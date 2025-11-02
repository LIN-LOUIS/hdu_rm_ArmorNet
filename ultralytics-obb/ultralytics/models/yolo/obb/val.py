# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.nms import TorchNMS


class OBBValidator(DetectionValidator):
    """
    面向旋转边界框(Oriented Bounding Box, OBB)模型的验证类,
    继承自 DetectionValidator,用于对预测旋转框的模型进行验证评估。

    该验证器专门用于评估预测旋转框的模型，常用于航空遥感、卫星图像等场景，
    因为这些图像中的目标方向各异。

    属性：
        args (dict): 验证器的配置参数。
        metrics (OBBMetrics): 用于评估 OBB 模型性能的度量对象。
        is_dota (bool): 指示验证数据集是否为 DOTA 格式。

    方法：
        init_metrics: 初始化 YOLO 的评估指标。
        _process_batch: 处理一批预测与真实框，计算 IoU 矩阵。
        _prepare_batch: 准备 OBB 验证批次数据。
        _prepare_pred: 对预测结果进行尺度和填充调整。
        plot_predictions: 在输入图像上绘制预测框。
        pred_to_json: 将 YOLO 预测结果序列化为 COCO JSON 格式。
        save_one_txt: 将预测结果保存为 txt 文件（归一化坐标）。
        eval_json: 以 JSON 格式评估 YOLO 输出并返回性能统计结果。

    示例：
        >>> from ultralytics.models.yolo.obb import OBBValidator
        >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml")
        >>> validator = OBBValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        初始化 OBBValidator 并将任务类型设置为 'obb'，度量对象设为 OBBMetrics。

        此构造函数用于创建一个 OBBValidator 实例，
        用于验证旋转边界框(OBB)模型。
        它扩展自 DetectionValidator 类，并针对 OBB 任务进行特定配置。

        参数：
            dataloader (torch.utils.data.DataLoader, 可选): 验证使用的数据加载器。
            save_dir (str | Path, 可选): 保存结果的目录。
            args (dict | SimpleNamespace, 可选): 包含验证参数的参数对象。
            _callbacks (list, 可选): 验证过程中的回调函数列表。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        初始化 YOLO-OBB 验证的评估指标。

        参数：
            model (torch.nn.Module): 待验证的模型。
        """
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # 验证集路径
        self.is_dota = isinstance(val, str) and "DOTA" in val  # 判断数据集是否为 DOTA 格式
        self.confusion_matrix.task = "obb"  # 设置混淆矩阵的任务类型为 'obb'

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        计算一批预测与真实框之间的正确匹配矩阵。

        参数：
            preds (dict[str, torch.Tensor]): 预测字典，包含 'cls'（类别）与 'bboxes'（预测框）。
            batch (dict[str, torch.Tensor]): 批次字典，包含真实的 'cls' 和 'bboxes'。

        返回：
            (dict[str, np.ndarray]): 包含 'tp' 键的字典，值为布尔矩阵，
            形状为 (N, 10)，表示每个检测在 10 个 IoU 阈值下的正确匹配情况。

        示例：
            >>> detections = torch.rand(100, 7)  # 100 个检测框
            >>> gt_bboxes = torch.rand(50, 5)    # 50 个真实框
            >>> gt_cls = torch.randint(0, 5, (50,))
            >>> correct_matrix = validator._process_batch(detections, gt_bboxes, gt_cls)
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        对模型的原始预测进行后处理。

        参数：
            preds (torch.Tensor): 模型的原始输出。

        返回：
            (list[dict[str, torch.Tensor]]): 处理后的预测结果，包含角度信息（angle）。
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # 拼接角度信息
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        为 OBB 验证准备单个批次的数据，进行适当的尺度与格式转换。

        参数：
            si (int): 当前批次索引。
            batch (dict[str, Any]): 批次数据，包含：
                - batch_idx: 批次索引张量
                - cls: 类别标签
                - bboxes: 边界框
                - ori_shape: 原始图像尺寸
                - img: 图像张量
                - ratio_pad: 尺度与填充信息

        返回：
            (dict[str, Any]): 处理后的批次数据（带缩放框与元信息）。
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # 调整目标框到图像尺寸
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def plot_predictions(self, batch: dict[str, Any], preds: list[torch.Tensor], ni: int) -> None:
        """
        在输入图像上绘制预测框并保存结果。

        参数：
            batch (dict[str, Any]): 包含图像、路径及元数据的批次。
            preds (list[torch.Tensor]): 每张图片的预测结果列表。
            ni (int): 当前批次编号，用于命名输出文件。

        示例：
            >>> validator = OBBValidator()
            >>> batch = {"img": images, "im_file": paths}
            >>> preds = [torch.rand(10, 7)]
            >>> validator.plot_predictions(batch, preds, 0)
        """
        for p in preds:
            # TODO: 修复重复调用 xywh2xyxy 的问题
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # 转换为 xyxy 坐标用于绘制
        super().plot_predictions(batch, preds, ni)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """
        将 YOLO 的预测结果转换为带旋转信息的 COCO JSON 格式。

        参数：
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 的预测结果。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad'、'im_file' 等信息的批次。

        说明：
            此方法将预测框同时转换为旋转框 (x, y, w, h, angle)
            与多边形框 (x1, y1, ..., x4, y4) 两种格式，并存入 JSON。
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = predn["bboxes"]
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for r, b, s, c in zip(rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "score": round(s, 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """
        将 YOLO OBB 检测结果保存为 txt 文件（归一化坐标）。

        参数：
            predn (torch.Tensor): 预测结果 (x, y, w, h, conf, cls, angle)。
            save_conf (bool): 是否保存置信度。
            shape (tuple[int, int]): 原始图像尺寸 (h, w)。
            file (Path): 输出文件路径。

        示例：
            >>> validator = OBBValidator()
            >>> predn = torch.tensor([[100, 100, 50, 30, 0.9, 0, 45]])
            >>> validator.save_one_txt(predn, True, (640, 480), "detection.txt")
        """
        import numpy as np
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放回原始图像尺寸。"""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"], predn["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """
        以 JSON 格式评估 YOLO 输出，并保存为 DOTA 格式文件。

        参数：
            stats (dict[str, Any]): 性能统计字典。

        返回：
            (dict[str, Any]): 更新后的性能统计结果。
        """
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # JSON 格式预测结果
            pred_txt = self.save_dir / "predictions_txt"    # DOTA 拆分结果
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))

            # 保存拆分后的结果
            LOGGER.info(f"正在将预测结果保存为 DOTA 格式到 {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

            # 保存合并后的结果（与官方脚本略有差异，Probiou 计算可能导致略低的 mAP）
            pred_merged_txt = self.save_dir / "predictions_merged_txt"
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"正在将合并预测结果保存为 DOTA 格式到 {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__", 1)[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)

            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # 类别偏移
                scores = bbox[:, 5]        # 置信度
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 使用 NMS 去重（阈值 0.3）
                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]
                    score = round(x[-2], 3)
                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
