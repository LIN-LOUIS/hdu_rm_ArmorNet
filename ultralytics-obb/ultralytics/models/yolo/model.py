# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """
    YOLO(You Only Look Once)目标检测模型。

    本类为 YOLO 模型提供统一接口，会根据模型文件名自动选择特定模型类型
    (如 YOLOWorld、YOLOE)，支持多种计算机视觉任务：
    - 目标检测(Detection)
    - 实例分割(Segmentation)
    - 分类(Classification)
    - 姿态估计(Pose Estimation)
    - 旋转边界框检测(OBB, Oriented Bounding Box)

    属性：
        model: 已加载的 YOLO 模型实例。
        task: 模型任务类型(detect, segment, classify, pose, obb)。
        overrides: 模型配置覆盖项。

    方法：
        __init__: 初始化 YOLO 模型，自动识别模型类型。
        task_map: 将任务映射到相应的模型、训练器、验证器和预测器。

    示例：
        加载 YOLOv11n 预训练检测模型：
        >>> model = YOLO("yolo11n.pt")

        加载 YOLO11n 分割模型：
        >>> model = YOLO("yolo11n-seg.pt")

        从 YAML 配置文件初始化：
        >>> model = YOLO("yolo11n.yaml")
    """

    def __init__(self, model: str | Path = "yolo11n.pt", task: str | None = None, verbose: bool = False):
        """
        初始化 YOLO 模型。

        构造函数会根据模型文件名自动识别类型(如 YOLOWorld、YOLOE)，
        并加载对应的网络结构与配置。

        参数：
            model (str | Path): 模型名称或路径，例如 'yolo11n.pt' 或 'yolo11n.yaml'。
            task (str, 可选): 指定任务类型(detect, segment, classify, pose, obb)，默认自动检测。
            verbose (bool): 是否在加载时显示模型信息。

        示例：
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")         # 加载检测模型
            >>> model = YOLO("yolo11n-seg.pt")     # 加载分割模型
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # YOLOWorld 模型
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # YOLOE 模型
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # 默认 YOLO 初始化
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # 检测 RT-DETR 结构
                from ultralytics import RTDETR
                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """定义任务类型与模型、训练器、验证器、预测器的映射关系。"""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {  # 旋转框任务
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """
    YOLO-World 开放词汇目标检测模型。

    YOLO-World 是一种“开放词汇检测”模型，
    可根据文本描述检测目标，而无需在特定类别上训练。
    它在 YOLO 架构上扩展了文本嵌入模块，
    支持实时开放词汇目标检测。

    属性：
        model: 已加载的 YOLO-World 模型实例。
        task: 固定为 'detect'。
        overrides: 模型配置覆盖项。

    示例：
        >>> model = YOLOWorld("yolov8s-world.pt")
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """
        初始化 YOLOv8-World 模型。

        参数：
            model (str | Path): 模型路径(支持 .pt / .yaml)。
            verbose (bool): 是否打印额外信息。
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # 如果没有自定义类别，则加载默认 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """任务映射：定义检测任务对应的类。"""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """
        设置模型检测类别。

        参数：
            classes (list[str]): 类别列表，例如 ["person", "car", "dog"]。
        """
        self.model.set_classes(classes)
        # 若包含背景类，则移除
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # 同步预测器类别名
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """
    YOLOE(Enhanced YOLO)模型。

    YOLOE 是 YOLO 的增强版，
    同时支持检测与实例分割，
    并引入视觉与文本位置嵌入、语义提示、跨模态对齐等特性。

    属性：
        model: 加载的 YOLOE 模型实例。
        task: 当前任务类型(detect 或 segment)。
        overrides: 模型配置覆盖项。

    示例：
        >>> model = YOLOE("yoloe-11s-seg.pt")
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        >>> results = model.predict("image.jpg")
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """初始化 YOLOE 模型。"""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """任务映射：检测与分割任务对应类。"""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """获取文本位置嵌入。"""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        获取图像特征的视觉位置嵌入。

        参数：
            img (torch.Tensor): 输入图像。
            visual (torch.Tensor): 视觉特征。

        返回：
            (torch.Tensor): 视觉位置嵌入。
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """
        设置模型的词汇表与类别名称。

        参数：
            vocab (list[str]): 模型使用的词汇。
            names (list[str]): 类别名称。
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """根据类别名获取词汇表。"""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """
        设置模型检测类别及其对应嵌入。

        参数：
            classes (list[str]): 类别列表。
            embeddings (torch.Tensor): 对应类别的嵌入(可选)。
        """
        assert isinstance(self.model, YOLOEModel)
        if embeddings is None:
            embeddings = self.get_text_pe(classes)
        self.model.set_classes(classes, embeddings)
        assert " " not in classes  # 不应包含背景类
        self.model.names = classes

        if self.predictor:
            self.predictor.model.names = classes

    def val(self, validator=None, load_vp: bool = False, refer_data: str | None = None, **kwargs):
        """
        使用文本或视觉提示进行验证。

        参数：
            validator (callable, 可选): 自定义验证函数。
            load_vp (bool): 是否加载视觉提示。
            refer_data (str, 可选): 引用数据路径。
        返回：
            dict: 验证指标。
        """
        custom = {"rect": not load_vp}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(self, source=None, stream: bool = False, visual_prompts: dict[str, list] = {},
                refer_image=None, predictor=yolo.yoloe.YOLOEVPDetectPredictor, **kwargs):
        """
        对图像、视频、目录或流进行预测。

        参数：
            source (str | int | np.ndarray): 输入源。
            stream (bool): 是否流式输出结果。
            visual_prompts (dict): 包含 'bboxes' 与 'cls' 的视觉提示。
            refer_image: 用作视觉提示参考的图像。
            predictor: 自定义预测器。
        返回：
            list | generator: 预测结果。
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"])
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides["imgsz"]),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None

        return super().predict(source, stream, **kwargs)
