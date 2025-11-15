# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# =============================
# 本文件为用户提供的 dataset.py 片段的「中文注释版」。
# 目标：在不改变任何功能与逻辑的前提下，补充中文行内注释，
#      便于快速理解 YOLO/OBB/多模态/grounding 数据集管线的实现细节。
# 说明：仅增加注释与极少量排版(空行)，不改动变量名与语义。
# =============================

from __future__ import annotations

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache 的版本号，需与当前代码期望的版本比对
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    用于加载 YOLO 标注(检测/分割/关键点/OBB)的通用数据集类。

    - use_segments: 是否使用语义/实例分割多边形(segments)
    - use_keypoints: 是否使用关键点(pose)
    - use_obb: 是否使用定向框(oriented bounding box)
    - data: 数据集配置字典(通常来自 data.yaml)
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        """
        Args:
            data: 数据集配置(含类别名、关键点形状等)
            task: 任务类型：'detect' | 'segment' | 'pose' | 'obb'
        """
        # 三个布尔开关根据 task 决定不同的读取/打包逻辑
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        # 分割与关键点不能同时为 True
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        # 调用父类，channels 从 data['channels'] 读取(默认 3)
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """
        扫描与校验图像与标签，生成缓存(含 shapes/hash/统计信息等)。
        - 使用 ThreadPool 并行调用 verify_image_label 提速。
        - 将每张图的 cls/bboxes/segments/keypoints 等组织为标准格式。
        - 最终写入 *.cache 以便下次快速加载。
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 统计：missing / found / empty / corrupt
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            # 关键点任务要求 data.yaml 中给定 kpt_shape=[K, D]
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        # 并行校验每张图像与其 label 文本
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f; nf += nf_f; ne += ne_f; nc += nc_f
                if im_file:
                    # 将一张图的标注整理为统一 dict 结构
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],      # [n,1]
                            "bboxes": lb[:, 1:],     # [n,4]，xywh 格式(归一化)
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        # 计算 hash 用于与缓存对比
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """
        加载/生成标签缓存，并做一致性检查与提示。
        - 若缓存存在且 hash/version 匹配，则直接用缓存。
        - 否则重新扫描，生成缓存。
        - 检查 boxes 与 segments 数量不匹配时给出警告并丢弃 segments。
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        # 进度条显示缓存摘要
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        # 提取 labels 列表并更新 im_files
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]

        # 统计是否为纯 boxes 或纯 segments，以避免 detect/segment 混合导致不一致
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        构建图像增强/预处理流水线。
        - 训练时：依据 mosaic/mixup/cutmix 等超参构造 v8_transforms
        - 验证/推理时：仅做 LetterBox 与 Format
        - Format 里会统一 bbox 格式、归一化、以及是否导出 mask/keypoint/obb
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # 仅训练阶段影响
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """在后期训练阶段关闭 mosaic/copy_paste/mixup/cutmix，稳定收敛。"""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """
        将原始标签字典转换为 Instances 对象以便后续组网/损失计算。
        - segments 若存在则进行等距重采样(非 OBB 情况下更多点数)
        - 最终生成 ultralytics.utils.instance.Instances
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # OBB 情况下 segment_resamples 置为更小(点数更少，避免误处理 OBB)
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # 若原始分割点数比目标 resamples 多，需重新插值保证等距
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # stack 成形状 [num_instances, segment_resamples, 2]
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        DataLoader 用的拼接函数：将不同样本的张量/列表按键合并为批次。
        - img/text_feats 用 torch.stack
        - visuals 用 pad_sequence(可变长)
        - masks/keypoints/bboxes/cls/segments/obb 用 cat
        - 维护 batch_idx，使得每个目标知道自己来自哪张图
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # 键排序，保证一致性
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # 让每个目标编号加上样本偏移
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    多模态数据集：在 YOLODataset 基础上，额外引入文本信息(类名同义词等)，
    以支持图文联合训练(如 grounding/对比学习等)。
    """

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """
        在父类 Instances 基础上，给每条样本补充 'texts' 字段：
        - data["names"] 的每个类别可以用 "/" 连接多个同义词，RandomLoadText 会随机选择其一。
        """
        labels = super().update_labels_info(label)
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """
        训练时插入 RandomLoadText，用于在线采样文本(包含负样本填充)。
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            # 插入到 Format 之前
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """返回类别名集合(拆分同义词，去空格)。"""
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}

    @property
    def category_freq(self):
        """统计每个类名(及同义词)在当前 labels 中出现的频次。"""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """选择高频词作为负样本填充的候选(上限 100)。"""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    基于 JSON(grounding 格式)读取标注的检测/分割数据集。
    - 与 YOLO 文本标注不同，这里从一个 JSON(包含 images/annotations)中解析得到。
    - 支持为每个 bbox 附带来自 caption 的文本片段，用于 grounding/短语定位训练。
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        # 仅支持 detect/segment 两种任务
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        # 这里 data 只设置了 channels=3，类别名由 JSON 解析时动态生成
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """覆写：图像文件列表在 get_labels 中由 JSON 动态读取，这里返回空列表。"""
        return []

    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        """
        可选的数据完整性验证：针对已知数据名，统计 bbox 实例数是否符合预期。
        未匹配的数据集会跳过。
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """
        从 JSON 读取 annotations，过滤 crowd/无效框，归一化 bbox，
        并将 segmentation(若有)转换为 boxes 或多边形点序列；同时抽取文本片段。
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                # COCO: [x,y,w,h] -> 中心点 xy + wh，并做归一化
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                # tokens_positive 给出 caption 的字符区间；拼接成类别名称
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    # 处理 segmentation：可能为多段，需要合并与归一化
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                # 若有多边形分割，则由 segments 反推 xywh 框(保持与 YOLO 接口一致)
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """优先从缓存加载；若缓存缺失/不匹配则重建。并做实例数验证与日志提示。"""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.json_file)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False
        [cache.pop(k) for k in ("hash", "version")]
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """与多模态类似：训练阶段插入 RandomLoadText，用于负采样与文本增强。"""
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """从 labels['texts'] 聚合所有文本类名(去重+strip)。"""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """统计文本类别的出现频次，用于负样本阈值。"""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """选择高频文本用于填充负样本(上限 100)。"""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    多数据集合并：将多个 YOLODataset 组成一个大的 Dataset，同时复用其 collate_fn。
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """沿用 YOLODataset 的 collate_fn。"""
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """批量关闭每个子数据集的 mosaic/mixup 等增强。"""
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: 支持语义分割专用数据集(目前占位)
class SemanticDataset(BaseDataset):
    """语义分割数据集占位类。"""

    def __init__(self):
        super().__init__()


class ClassificationDataset:
    """
    图像分类数据集：基于 torchvision 的 ImageFolder，
    额外支持：
    - 可选的数据缓存(RAM/磁盘 *.npy)
    - 训练增强(AutoAugment/随机擦除/HFlip/VFlip/HSV 等)
    - 图像合法性快速校验与 *.cache 存取
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        import torchvision  # 延迟导入以加速 ultralytics 的整体 import

        # torchvision 0.18+ 支持 allow_empty，避免空类报错
        if TORCHVISION_0_18:
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples  # [(filepath, class_idx), ...]
        self.root = self.base.root

        # 训练抽样(fraction < 1.0 可减少样本量用于快速实验)
        if augment and args.fraction < 1.0:
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""

        # 缓存策略：RAM 或磁盘(注意：RAM 模式存在历史内存泄漏 issue，这里强制关闭)
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"

        # 先做图像合法性校验，返回过滤后的 samples
        self.samples = self.verify_images()
        # 为每个样本记录对应的 npy 路径与缓存图像位(RAM模式)
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]

        # 构建增强/预处理流水线
        scale = (1.0 - args.scale, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> dict:
        """
        读取第 i 个样本：
        - 依据缓存策略选择读取源(RAM/磁盘 *.npy/直接 cv2.imread)
        - 转为 PIL，再走 torchvision 的 transforms
        - 返回 {"img": tensor, "cls": class_index}
        """
        f, j, fn, im = self.samples[i]  # 文件名、类索引、npy 路径、RAM 缓存图像
        if self.cache_ram:
            if im is None:
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)  # BGR
        # OpenCV -> PIL(RGB)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        return len(self.samples)

    def verify_images(self) -> list[tuple]:
        """
        扫描并过滤不可读/损坏图像；使用 *.cache 加速下次启动。
        - check_file_speeds：抽样测速、快速探测 I/O 问题
        - load/save_dataset_cache_file：读取/写入缓存(含哈希与数量)
        - 返回过滤后的 [(filepath, class_idx), ...]
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache 路径

        try:
            # 尝试读取缓存并校验版本与哈希
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)
            cache = load_dataset_cache_file(path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([x[0] for x in self.samples])
            nf, nc, n, samples = cache.pop("results")
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # 缓存不可用，则逐个校验
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
