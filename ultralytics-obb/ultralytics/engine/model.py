# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Ultralytics 开源协议 AGPL-3.0 - https://ultralytics.com/license

from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import guess_model_task, load_checkpoint, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    YAML,
    callbacks,
    checks,
)


class Model(torch.nn.Module):
    """
    YOLO 模型的基础类，为不同类型的模型提供统一接口。

    该类封装了 YOLO 系列模型的通用功能，包括训练、验证、推理、导出和基准测试等。
    它支持从本地文件、Ultralytics HUB 云端或 Triton Server 加载模型，并自动适配任务类型。

    主要属性：
        callbacks: 回调函数字典，用于在模型生命周期中触发自定义事件。
        predictor: 用于执行推理的预测器对象。
        model: 实际的 PyTorch 模型实例。
        trainer: 训练器对象，用于模型训练。
        ckpt: 如果从 .pt 文件加载模型，则此字段存储检查点内容。
        cfg: 若从 .yaml 文件加载模型，则此字段为模型配置文件路径。
        ckpt_path: 模型检查点文件路径。
        overrides: 训练/推理配置参数的覆盖字典。
        metrics: 最新训练或验证指标。
        session: 若模型来自 HUB，则为当前的云端训练会话。
        task: 模型任务类型(检测、分割、分类、姿态估计、旋转框检测等）。
        model_name: 模型名称。

    示例：
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: str | Path | Model = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        初始化 YOLO 模型实例。

        本函数根据输入路径或模型名称加载 YOLO 模型，支持本地文件、Ultralytics HUB 模型和 Triton Server 模型。
        初始化后可直接进行训练、推理或导出操作。

        参数：
            model: 模型路径或名称，可为 .pt 权重文件、.yaml 配置文件、HUB 模型 URL 或 Triton 模型。
            task: 模型任务类型(可选），若未指定则自动推断。
            verbose: 是否显示详细日志。

        异常：
            FileNotFoundError: 指定的模型文件不存在。
            ValueError: 文件格式不受支持。
            ImportError: 缺少必要依赖。
        """
        if isinstance(model, Model):
            # 若传入的参数已是 Model 对象，则直接复制其属性
            self.__dict__ = model.__dict__
            return

        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()  # 初始化默认回调
        self.predictor = None  # 预测器对象
        self.model = None  # 模型本体
        self.trainer = None  # 训练器对象
        self.ckpt = {}  # 若从 .pt 加载模型，则此处保存检查点内容
        self.cfg = None  # 若从 .yaml 加载模型，则此处保存配置路径
        self.ckpt_path = None  # 检查点路径
        self.overrides = {}  # 覆盖参数
        self.metrics = None  # 性能指标
        self.session = None  # HUB 会话
        self.task = task  # 模型任务类型
        self.model_name = None  # 模型名称

        model = str(model).strip()

        # 如果是 Ultralytics HUB 模型(形如 https://hub.ultralytics.com/models/...）
        if self.is_hub_model(model):
            from ultralytics.hub import HUBTrainingSession

            checks.check_requirements("hub-sdk>=0.0.12")  # 确保 hub-sdk 已安装
            session = HUBTrainingSession.create_session(model)
            model = session.model_file  # 下载模型文件
            if session.train_args:  # 若为 HUB 训练任务
                self.session = session

        # 如果是 Triton Server 模型
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # 默认检测任务
            return

        # 启用确定性 CUDA 行为，避免运行时警告
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # 若以 YAML 文件结尾，则创建新模型
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            # 否则加载已训练模型
            self._load(model, task=task)

        # 删除父类的 training 属性，以便直接访问 self.model.training
        del self.training

    def __call__(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        允许模型对象被直接调用以执行推理。

        该方法等价于 predict()，可直接通过 `model(source)` 调用。

        参数：
            source: 输入源，可为图片路径、PIL 图像、numpy 数组、torch.Tensor 或视频流等。
            stream: 是否以流模式推理(返回生成器）。
            kwargs: 传递给 predict() 的额外参数。

        返回：
            list[Results]: 包含推理结果的对象列表。
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        判断给定字符串是否为 Triton Server 模型地址。

        参数：
            model: 待检测字符串。

        返回：
            bool: 若为合法 Triton URL(http/grpc），返回 True。
        """
        from urllib.parse import urlsplit
        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        判断是否为 Ultralytics HUB 模型。

        参数：
            model: 模型字符串。

        返回：
            bool: 若为合法 HUB 模型 URL，返回 True。
        """
        from ultralytics.hub import HUB_WEB_ROOT
        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        创建新模型，并根据配置文件推断任务类型。

        该函数用于从 .yaml 模型定义文件初始化模型结构，支持自动识别任务类型并加载对应的网络结构。

        参数：
            cfg: 模型配置文件路径(YAML 格式）。
            task: 指定任务类型，若为空则自动推断。
            model: 若传入自定义模型实例，则直接使用。
            verbose: 是否输出详细日志。
        """
        cfg_dict = yaml_model_load(cfg)  # 读取 YAML 配置
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)  # 若未指定 task，则根据配置推断

        # 使用 _smart_load 加载相应任务的模型类并实例化
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # 合并默认配置和模型参数，便于导出
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg
    def _load(self, weights: str, task=None) -> None:
        """
        从检查点文件加载模型或初始化权重。

        该函数支持从 .pt 检查点或其他权重文件加载模型，并设置任务类型与模型参数。

        参数：
            weights: 模型权重文件路径。
            task: 指定任务类型，若为空则自动推断。

        异常：
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式不支持。
        """
        # 若输入为网络链接(HTTP / RTSP / RTMP 等），则先下载
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])
        # 检查文件合法性(补全后缀 .pt）
        weights = checks.check_model_file_from_stem(weights)

        # 若为 PyTorch 检查点文件
        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            # 其他类型文件直接加载
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights

        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        检查当前模型是否为 PyTorch 模型。

        若模型不是 torch.nn.Module 或 .pt 文件，则抛出异常。
        某些操作(如训练、验证、导出）仅在 PyTorch 模型下可执行。
        """
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"当前模型 '{self.model}' 不是有效的 PyTorch 模型。仅支持 .pt 文件或 torch.nn.Module 对象。"
                f"ONNX、TensorRT 等导出格式仅可执行 predict/val，而非 train/export。"
            )

    def reset_weights(self) -> Model:
        """
        重置模型参数为初始状态。

        遍历模型的所有模块，如果模块存在 reset_parameters() 方法，则执行重置。
        同时确保所有参数的 requires_grad = True。

        返回：
            self：重置后的模型对象。
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: str | Path = "yolo11n.pt") -> Model:
        """
        从指定权重文件加载参数。

        参数：
            weights: 权重文件路径。

        返回：
            self：加载权重后的模型对象。
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """
        保存模型当前状态到指定路径。

        参数：
            filename: 保存路径(默认 saved_model.pt）
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        显示模型结构信息。

        参数：
            detailed: 若为 True，显示详细层级参数信息。
            verbose: 若为 False，返回信息字符串列表而非打印输出。

        返回：
            模型结构与参数信息(若 verbose=False）。
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        """
        将 Conv2d 与 BatchNorm2d 层融合以加速推理。

        该过程通过折叠 BN 参数(均值、方差、权重、偏置）进卷积层，
        从而减少推理时的运算量与显存访问次数。
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        基于输入源生成图像嵌入特征。

        参数：
            source: 输入源(图片路径、PIL 图像、numpy 数组等）。
            stream: 是否以流方式处理。
            kwargs: 其他可选参数。

        返回：
            list[torch.Tensor]：生成的图像特征张量。
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # 默认取倒数第二层
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        """
        执行推理操作。

        参数：
            source: 输入源，可为文件路径、图像数组或视频流。
            stream: 是否以流模式返回生成器。
            predictor: 自定义预测器对象(可选）。
            kwargs: 其他自定义推理参数，如 conf、device、half 等。

        返回：
            list[Results]：推理结果列表。
        """
        # 若未指定输入，则默认使用 Ultralytics 示例图片
        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' 未指定，默认使用 {source}")

        # 判断是否从命令行调用
        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        # 默认推理参数
        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)  # 用于 SAM 模型提示词

        # 若 predictor 未初始化，则创建新的预测器
        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            # 若已存在预测器，则仅更新参数
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)

        # 若为 SAM 模型，则加载提示词
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)

        # 返回结果
        if is_cli:
            return self.predictor.predict_cli(source=source)
        else:
            gen = self.predictor.stream_inference(source=source)
            return gen if stream else list(gen)

    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[Results]:
        """
        对输入源进行目标跟踪。

        该方法使用注册的跟踪器在视频、实时流或图像序列中进行目标跟踪。
        支持多种跟踪算法(如 ByteTrack）。

        参数：
            source: 输入源(文件路径、URL、视频流等）。
            stream: 是否以流模式进行推理。
            persist: 是否在多次调用中保留跟踪状态。
            kwargs: 其他自定义参数。

        返回：
            list[Results]：包含跟踪结果的结果对象列表。
        """
        # 若当前 predictor 未注册 tracker，则注册
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker
            register_tracker(self, persist)

        # ByteTrack 需要较低置信度输入
        kwargs["conf"] = kwargs.get("conf") or 0.1
        kwargs["batch"] = kwargs.get("batch") or 1  # 视频默认 batch=1
        kwargs["mode"] = "track"

        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        对模型进行验证。

        支持自定义验证器(validator），或默认使用任务对应的验证逻辑。
        可自定义数据集路径、输入尺寸、设备等。

        参数：
            validator: 自定义验证器类。
            kwargs: 其他验证参数(data、imgsz、device、batch 等）。

        返回：
            验证指标(如 mAP、F1-score 等）。
        """
        custom = {"rect": True}  # 默认矩形推理
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        """
        对模型进行基准性能测试(Benchmark）。

        该函数将模型导出为不同格式(如 ONNX、TensorRT、CoreML）并评估各自推理速度与精度。

        参数：
            data: 数据集路径(默认 None 表示使用内置数据）。
            format: 指定导出格式进行单一基准评测(如 'onnx'）。
            verbose: 是否输出详细日志。
            kwargs: 其他参数(如 imgsz, half, int8, device 等）。

        返回：
            dict：各格式模型的性能指标(推理时间、mAP、参数量等）。
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark
        from .exporter import export_formats

        custom = {"verbose": False}
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}

        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format, [])) - {"batch"}
        export_kwargs = {k: v for k, v in args.items() if k in export_args}

        return benchmark(
            model=self,
            data=data,
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            **export_kwargs,
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        导出模型至多种格式以便部署。

        支持导出到 ONNX、TensorRT、TorchScript、CoreML、OpenVINO 等主流格式，
        可选择半精度 (FP16)、整型量化 (INT8)、动态尺寸等模式。

        参数：
            kwargs: 导出参数，例如：
                format: 导出格式(如 'onnx'、'engine'、'coreml'）
                half: 是否使用半精度
                int8: 是否使用 INT8 量化
                device: 导出设备
                simplify: 是否简化 ONNX 图
                workspace: TensorRT 最大显存分配
                nms: 是否添加非极大值抑制模块

        返回：
            str：导出文件的路径。
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}

        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)
    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        使用指定的数据集与配置进行模型训练。

        支持自定义训练器(trainer）或默认的 YOLO 训练器。
        当模型连接 Ultralytics HUB 云端时，会自动同步训练会话参数。
        支持从 checkpoint 恢复训练、修改参数、或自定义优化器。

        参数：
            trainer: 自定义训练器类(可选）。
            kwargs: 训练相关配置，例如：
                - data: 数据集配置文件路径
                - epochs: 训练轮数
                - batch: 批次大小
                - imgsz: 输入图像大小
                - device: 运行设备(cuda / cpu）
                - optimizer: 优化器类型(如 SGD / AdamW）
                - lr0: 初始学习率
                - patience: 早停轮数(未提升自动终止训练）
                - resume: 是否从上次 checkpoint 恢复训练

        返回：
            metrics: 若训练成功则返回验证指标(如 mAP、Precision、Recall）。
        """
        self._check_is_pytorch_model()

        # 若存在 Ultralytics HUB 会话，则优先使用其云端配置
        if hasattr(self.session, "model") and self.session.model.id:
            if any(kwargs):
                LOGGER.warning("检测到 HUB 远程训练，已忽略本地训练参数。")
            kwargs = self.session.train_args

        # 检查 pip 是否有新版本(防止依赖不一致）
        checks.check_pip_update_available()

        # 若传入 pretrained 参数，则加载权重
        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            self.load(kwargs["pretrained"])

        # 若 cfg 存在，则加载配置文件；否则使用 overrides
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides

        # 默认参数：确保 data、model、task 一致
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }

        # 合并配置：优先级从左到右(默认 < overrides < kwargs）
        args = {**overrides, **custom, **kwargs, "mode": "train", "session": self.session}

        # 若启用 resume 模式，则从 ckpt_path 继续训练
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        # 加载训练器
        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)

        # 若不是 resume 模式，则构建新模型
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        # 开始训练
        self.trainer.train()

        # 训练结束后更新模型与指标
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.metrics = getattr(self.trainer.validator, "metrics", None)

        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        进行超参数调优(Hyperparameter Tuning）。

        支持两种模式：
            1. 内置 Tuner(默认）
            2. Ray Tune(分布式搜索）

        参数：
            use_ray: 是否使用 Ray Tune 调参(True 为分布式模式）。
            iterations: 调参迭代次数。
            *args, **kwargs: 其他传入参数，如：
                - data: 数据集路径
                - epochs: 每次实验训练轮数
                - lr0, momentum, weight_decay 等优化参数范围

        返回：
            dict：最佳参数及对应指标结果。
        """
        self._check_is_pytorch_model()

        # 若启用 Ray Tune 模式
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)

        # 否则使用内置调参器
        else:
            from .tuner import Tuner
            custom = {}
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)
    def _apply(self, fn) -> Model:
        """
        将给定函数应用到模型的所有张量(包括非参数张量）。

        用于模型设备迁移、精度变换(如 .to('cuda')、.half()）等操作。
        同时会重置 predictor(防止设备变更导致的上下文错误），
        并更新 overrides 中的 device 参数。

        参数：
            fn: 待应用的函数，如 lambda t: t.cuda()。

        返回：
            Model: 应用函数后的模型实例。
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None  # 重置预测器(设备可能已改变）
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> dict[int, str]:
        """
        获取当前模型的类别名称映射。

        返回模型的类别字典(index -> name）。
        若 predictor 尚未初始化，会先自动加载 predictor。

        返回：
            dict: {类别索引: 类别名称}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)

        if not self.predictor:
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)
            return predictor.model.names

        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        获取当前模型所在设备(CPU 或 GPU）。

        返回：
            torch.device: 当前模型参数所在设备。
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """
        获取模型使用的数据预处理(transforms）。

        返回：
            transforms 对象(若存在）或 None。
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        为指定事件注册新的回调函数。

        回调函数用于在训练/验证/推理过程的特定阶段执行自定义逻辑。
        例如 on_train_start、on_epoch_end、on_predict_end 等。

        参数：
            event: 事件名称(字符串，如 'on_train_start'）。
            func: 回调函数。

        示例：
            >>> def on_train_start(trainer): print("训练开始！")
            >>> model.add_callback("on_train_start", on_train_start)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        清除指定事件的所有回调函数。

        该方法会将某事件下的所有自定义及默认回调全部移除。

        参数：
            event: 要清除的事件名称。

        示例：
            >>> model.clear_callback("on_train_start")
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        重置所有回调为默认设置。

        将所有事件的回调函数恢复为 Ultralytics 框架内置的默认回调。
        当进行了大量自定义回调调试后，可用此方法回到原始状态。
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        """
        在加载 PyTorch 模型 checkpoint 时重置部分参数。

        仅保留关键字段(imgsz、data、task、single_cls），
        避免旧 checkpoint 参数影响新训练。

        参数：
            args: 原始 checkpoint 参数字典。

        返回：
            dict: 精简后的参数字典。
        """
        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """
        智能加载模型组件(如 model、trainer、validator、predictor）。

        根据当前任务(detect、segment、pose、obb 等）自动匹配正确的模块。
        若该任务类型不支持对应模式，会抛出异常。

        参数：
            key: 模块类型('model' / 'trainer' / 'validator' / 'predictor'）

        返回：
            对应模块类。

        异常：
            NotImplementedError: 若当前任务不支持该模块。
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(f"模型 '{name}' 不支持 '{self.task}' 任务下的 '{mode}' 模式。") from e

    @property
    def task_map(self) -> dict:
        """
        定义任务到模块(model/trainer/validator/predictor）的映射关系。

        每个任务(detect、segment、classify、pose、obb 等）
        对应一组实现类，框架通过该映射动态加载合适的组件。

        返回：
            dict[str, dict[str, Any]]: 任务名 -> 模块映射。
        """
        raise NotImplementedError("请在子类中定义 task_map 映射！")

    def eval(self):
        """
        将模型切换为评估模式(evaluation mode）。

        评估模式下，模型会禁用 dropout，并固定 BatchNorm 均值方差。
        用于推理阶段的稳定输出。

        返回：
            Model: 已设置为 eval 模式的模型。
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        允许直接通过 Model 实例访问底层模型的属性。

        若访问属性名为 'model'，则返回 self._modules['model']；
        否则直接代理到 self.model 的对应属性。

        参数：
            name: 属性名。

        返回：
            对应属性的值。

        示例：
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            >>> print(model.stride)
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)

