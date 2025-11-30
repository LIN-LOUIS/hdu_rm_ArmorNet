# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    用于加载和处理图像数据的基础数据集类。

    该类提供在目标检测任务中加载图像、缓存图像，以及为训练与推理准备数据的核心功能。

    属性说明：
        img_path (str): 图像文件夹路径。
        imgsz (int): 图像缩放后的目标尺寸。
        augment (bool): 是否进行数据增强。
        single_cls (bool): 是否将所有目标视为单一类别。
        prefix (str): 日志打印前缀。
        fraction (float): 使用的数据集比例。
        channels (int): 图像通道数(灰度图为 1，RGB 图为 3)。
        cv2_flag (int): OpenCV 读取图像的标志位。
        im_files (list[str]): 图像文件路径列表。
        labels (list[dict]): 标签字典列表。
        ni (int): 数据集中图像的数量。
        rect (bool): 是否使用矩形训练(rectangular training)。
        batch_size (int): batch 大小。
        stride (int): 模型使用的步长。
        pad (float): 填充系数。
        buffer (list): 用于 mosaic 增强的缓冲区。
        max_buffer_length (int): 缓冲区最大长度。
        ims (list): 已加载图像列表。
        im_hw0 (list): 原始图像尺寸列表 (h, w)。
        im_hw (list): 缩放后图像尺寸列表 (h, w)。
        npy_files (list[Path]): numpy 文件路径列表。
        cache (str): 训练时图像缓存到内存或磁盘的方式。
        transforms (callable): 图像变换函数。
        batch_shapes (np.ndarray): 矩形训练下每个 batch 的图像尺寸。
        batch (np.ndarray): 每张图像对应的 batch 索引。

    方法：
        get_img_files: 从指定路径中读取图像文件。
        update_labels: 根据给定类别列表筛选标签。
        load_image: 从数据集中加载一张图像。
        cache_images: 将图像缓存到内存或磁盘。
        cache_images_to_disk: 将图像保存为 *.npy 文件以加速加载。
        check_cache_disk: 检查磁盘空间是否满足缓存需求。
        check_cache_ram: 检查内存是否满足缓存需求。
        set_rectangle: 为矩形训练设置每个 batch 的图像尺寸。
        get_image_and_label: 获取图像和对应标签信息。
        update_labels_info: 自定义标签格式(由子类实现)。
        build_transforms: 构建数据增强与变换流程(由子类实现)。
        get_labels: 获取标签数据(由子类实现)。
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """
        使用给定配置和选项初始化 BaseDataset。

        参数：
            img_path (str | list[str]): 图像文件夹路径或图像路径列表。
            imgsz (int): 图像缩放后的尺寸。
            cache (bool | str): 训练时是否将图像缓存到内存或磁盘。
            augment (bool): 若为 True，则启用数据增强。
            hyp (dict[str, Any]): 用于控制数据增强的超参数。
            prefix (str): 日志打印前缀。
            rect (bool): 若为 True，则启用矩形训练。
            batch_size (int): batch 大小。
            stride (int): 模型的步长。
            pad (float): 填充系数。
            single_cls (bool): 若为 True，则视为单类别训练。
            classes (list[int], optional): 需要保留的类别列表。
            fraction (float): 数据集中使用的样本比例。
            channels (int): 图像通道数(灰度 1，RGB 3)。
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls 与 include_class 共同生效
        self.ni = len(self.labels)  # 图像数量
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # mosaic 增强的缓冲区(容量默认为 batch 大小)
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 图像缓存设置(cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' 可能导致训练结果非确定性。若磁盘空间允许，可考虑使用 cache='disk' 以获得确定性训练结果。"
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # 构建图像变换与数据增强流程
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """
        从指定路径中读取所有图像文件。

        参数：
            img_path (str | list[str]): 图像目录或文件路径(或其列表)。

        返回：
            (list[str]): 图像文件路径列表。

        异常：
            FileNotFoundError: 当找不到任何图像或路径不存在时抛出。
        """
        try:
            f = []  # 图像文件列表
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # 跨平台路径
                if p.is_dir():  # 目录
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # 也可以使用：F = list(p.rglob('*.*'))  # pathlib 写法
                elif p.is_file():  # 文件
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        # 将相对路径转换为绝对路径
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
                        # 也可以使用 pathlib：
                        # F += [p.parent / x.lstrip(os.sep) for x in t]
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} 不存在")
            # 仅保留图像格式，并统一分隔符
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # pathlib 对应写法：
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])
            assert im_files, f"{self.prefix}在 {img_path} 中未找到任何图像文件。{FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}从 {img_path} 加载数据时出错\n{HELP_URL}") from e
        if self.fraction < 1:
            # 仅保留数据集的一部分
            im_files = im_files[: round(len(im_files) * self.fraction)]
        # 检查图像读取速度
        check_file_speeds(im_files, prefix=self.prefix)
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """
        根据指定类别列表更新标签，仅保留需要的类别。

        参数：
            include_class (list[int], optional): 要保留的类别列表。若为 None，则保留所有类别。
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                # 根据类别筛选
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                # 若为单类别训练，则强制将类别设为 0
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        从数据集索引 i 处加载一张图像。

        参数：
            i (int): 图像索引。
            rect_mode (bool): 是否使用“保持长边不超过 imgsz 的等比例缩放”。

        返回：
            im (np.ndarray): 加载后的图像数组。
            hw_original (tuple[int, int]): 原始图像尺寸 (h, w)。
            hw_resized (tuple[int, int]): 缩放后图像尺寸 (h, w)。

        异常：
            FileNotFoundError: 当图像文件不存在时抛出。
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # 未缓存到内存中
            if fn.exists():  # 若有 npy 缓存，则优先加载
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}检测到损坏的 *.npy 文件 {fn}，已删除，错误原因：{e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # 直接读取图像
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"找不到图像文件 {f}")

            h0, w0 = im.shape[:2]  # 原始高宽
            if rect_mode:  # 按长边等比例缩放到 imgsz，保持长宽比
                r = self.imgsz / max(h0, w0)  # 缩放比例
                if r != 1:  # 尺寸不相等才缩放
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # 否则直接拉伸到方形 imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                # 灰度图扩展为 H×W×1
                im = im[..., None]

            # 若正在使用数据增强，则将图像加入缓冲区
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, 原始尺寸, 缩放尺寸
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # 防止缓冲区为空
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        # 若已缓存，则直接返回
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """将图像缓存到内存或磁盘，以加速训练。"""
        b, gb = 0, 1 << 30  # 已缓存图像的字节数，每 GB 的字节数
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    # im, 原始尺寸, 缩放尺寸 = load_image(self, i)
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}正在缓存图像 ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """将图像保存为 *.npy 文件，以便下次快速加载。"""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """
        检查磁盘空间是否足够缓存所有图像。

        参数：
            safety_margin (float): 安全冗余系数，用于对磁盘空间进行放大预估。

        返回：
            (bool): 若磁盘空间足够则返回 True，否则返回 False。
        """
        import shutil

        b, gb = 0, 1 << 30  # 已估算缓存大小，每 GB 的字节数
        n = min(self.ni, 30)  # 从最多 30 张图像中估算
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}图像目录不可写，跳过磁盘缓存")
                return False
        # 估算完整数据集缓存到磁盘所需空间
        disk_required = b * self.ni / n * (1 + safety_margin)
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}缓存数据集大约需要 {disk_required / gb:.1f}GB 磁盘空间，"
                f"安全冗余为 {int(safety_margin * 100)}%，但当前仅有 "
                f"{free / gb:.1f}/{total / gb:.1f}GB 可用，故不启用磁盘缓存"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """
        检查内存是否足够将图像缓存到 RAM。

        参数：
            safety_margin (float): 安全冗余系数，用于对内存空间进行放大预估。

        返回：
            (bool): 若内存空间足够则返回 True，否则返回 False。
        """
        b, gb = 0, 1 << 30  # 已估算缓存大小，每 GB 的字节数
        n = min(self.ni, 30)  # 从最多 30 张图像中估算
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # 抽样一张图像
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # 按最大边缩放的比例
            b += im.nbytes * ratio**2
        # 估算完整数据集缓存到内存所需空间
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}将图像缓存到内存大约需要 {mem_required / gb:.1f}GB，"
                f"安全冗余为 {int(safety_margin * 100)}%，但当前仅有 "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB 可用，故不启用内存缓存"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """为 YOLO 检测的矩形训练模式设置每个 batch 的图像尺寸。"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # 每张图像对应的 batch 索引
        nb = bi[-1] + 1  # batch 总数

        s = np.array([x.pop("shape") for x in self.labels])  # 每张图像的高宽 (h, w)
        ar = s[:, 0] / s[:, 1]  # 高宽比
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # 为每个 batch 设置训练图像的目标形状
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        # 根据 imgsz、stride 与 pad 计算最终的 batch_shapes
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # 每张图像对应的 batch 索引

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回给定索引对应的图像与标签，并应用变换。"""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """
        获取指定索引处的图像和标签信息。

        参数：
            index (int): 图像索引。

        返回：
            (dict[str, Any]): 包含图像和元信息的标签字典。
        """
        # 需要使用 deepcopy，避免在原标签上直接修改
        # 参考：https://github.com/ultralytics/ultralytics/pull/1948
        label = deepcopy(self.labels[index])
        label.pop("shape", None)  # shape 仅用于矩形训练，这里删除
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # 评估时用于尺寸还原
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """返回数据集中标签列表的长度(即图像数量)。"""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """可在此处自定义标签格式，由子类按需重写。"""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """
        用户可在此自定义数据增强和变换流程。

        示例：
            >>> if self.augment:
            ...     # 训练时的变换
            ...     return Compose([])
            >>> else:
            ...     # 验证时的变换
            ...     return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """
        用户可在此自定义标签的读取与格式。

        示例：
            输出应为包含以下键的字典：
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # 图像尺寸 (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # 多边形坐标 xy
            ...     keypoints=keypoints,  # 关键点坐标 xy
            ...     normalized=True,  # 或 False
            ...     bbox_format="xyxy",  # 或 xywh, ltwh
            ... )
        """
        raise NotImplementedError
