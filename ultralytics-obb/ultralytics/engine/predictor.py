# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
在图像、视频、目录、通配符、YouTube、摄像头、流媒体等上运行推理。

使用示例 - 输入源:
    $ yolo mode=predict model=yolo11n.pt source=0                               # 摄像头
                                                img.jpg                         # 单张图像
                                                vid.mp4                         # 视频文件
                                                screen                          # 屏幕截图
                                                path/                           # 图像文件夹
                                                list.txt                        # 图像路径列表
                                                list.streams                    # 流媒体列表
                                                'path/*.jpg'                    # 通配符路径
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube 视频
                                                'rtsp://example.com/media.mp4'  # RTSP、RTMP、HTTP 或 TCP 流

使用示例 - 模型格式:
    $ yolo mode=predict model=yolo11n.pt                 # PyTorch 格式
                              yolo11n.torchscript        # TorchScript 格式
                              yolo11n.onnx               # ONNX Runtime 或 OpenCV DNN(使用 dnn=True)
                              yolo11n_openvino_model     # OpenVINO
                              yolo11n.engine             # TensorRT
                              yolo11n.mlpackage          # CoreML(仅 macOS)
                              yolo11n_saved_model        # TensorFlow SavedModel
                              yolo11n.pb                 # TensorFlow GraphDef
                              yolo11n.tflite             # TensorFlow Lite
                              yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo11n_paddle_model       # PaddlePaddle
                              yolo11n.mnn                # MNN
                              yolo11n_ncnn_model         # NCNN
                              yolo11n_imx_model          # Sony IMX
                              yolo11n_rknn_model         # Rockchip RKNN
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode
from ultralytics.engine.results import Results
from ultralytics.models.digit_classifier import classify_rois
STREAM_WARNING = """
如果未传入 `stream=True` 参数，推理结果会不断累积在内存(RAM)中，
对于较大的输入源或长时间运行的视频流，可能会导致内存溢出。
详情请参考：https://docs.ultralytics.com/modes/predict/

示例:
    results = model(source=..., stream=True)  # 结果生成器
    for r in results:
        boxes = r.boxes  # 检测框对象
        masks = r.masks  # 分割掩码对象
        probs = r.probs  # 分类概率对象
"""


class BasePredictor:
    """
    基础预测器类。

    该类为各种推理任务提供基础功能，包括模型加载、推理执行和结果处理，
    可适用于多种输入源(图像、视频、流媒体等)。

    属性:
        args (SimpleNamespace): 推理配置参数。
        save_dir (Path): 保存结果的路径。
        done_warmup (bool): 模型是否完成预热。
        model (torch.nn.Module): 用于推理的模型。
        data (dict): 数据配置。
        device (torch.device): 推理使用的设备。
        dataset (Dataset): 用于推理的数据集对象。
        vid_writer (dict[str, cv2.VideoWriter]): 视频输出写入器，键为保存路径。
        plotted_img (np.ndarray): 最近一次绘制的图像。
        source_type (SimpleNamespace): 输入源类型。
        seen (int): 已处理的图像数量。
        windows (list[str]): 用于显示的窗口列表。
        batch (tuple): 当前批次数据。
        results (list[Any]): 当前批次推理结果。
        transforms (callable): 图像预处理转换函数。
        callbacks (dict[str, list[callable]]): 各事件的回调函数。
        txt_path (Path): 文本结果保存路径。
        _lock (threading.Lock): 多线程安全锁。

    方法:
        preprocess: 推理前对输入图像进行预处理。
        inference: 运行模型推理。
        postprocess: 对原始预测结果进行后处理。
        predict_cli: 在命令行模式下运行推理。
        setup_source: 设置输入源及推理模式。
        stream_inference: 在流媒体上实时推理。
        setup_model: 初始化并配置模型。
        write_results: 将推理结果写入文件。
        save_predicted_images: 保存带预测结果的可视化图像。
        show: 显示结果。
        run_callbacks: 运行事件回调。
        add_callback: 添加新的回调函数。
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, list[callable]] | None = None,
    ):
        """
        初始化 BasePredictor 类。

        参数:
            cfg (str | dict): 配置文件路径或配置字典。
            overrides (dict, 可选): 配置项覆盖。
            _callbacks (dict, 可选): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认置信度阈值 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 初始化默认变量(在 setup 后才能使用)
        self.model = None
        self.data = self.args.data  # 数据配置字典
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # 视频写入器字典 {save_path: writer}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # 确保多线程推理安全
        callbacks.add_integration_callbacks(self)
        #添加 ROI 缓存属性
        self.armor_rois = []
        def _rbox_to_quad(self, cx, cy, w, h, angle_rad):
            """将 (cx, cy, w, h, theta[rad]) 转为四边形四点，顺序：lt, rt, rb, lb"""
            cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
            dx, dy = w / 2.0, h / 2.0
            corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            rot = corners @ R.T
            rot[:, 0] += cx
            rot[:, 1] += cy
            return rot.astype(np.float32)

        def _crop_quad(self, im0, quad, out_size=(64, 64)):
            """将任意四边形 ROI 透视到固定小图（默认 64x64）"""
            dst = np.array([[0, 0], [out_size[0]-1, 0], [out_size[0]-1, out_size[1]-1], [0, out_size[1]-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            roi = cv2.warpPerspective(im0, M, out_size)
            return roi
 
    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """
        在推理前对输入图像进行预处理。

        参数:
            im (torch.Tensor | list[np.ndarray]): 输入图像，
                若为张量则形状为 (N, 3, H, W)，
                若为列表则为 [(H, W, 3) × N]。

        返回:
            (torch.Tensor): 预处理后的图像张量，形状为 (N, 3, H, W)。
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR 转 RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC → BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # 保证内存连续
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 → fp16/32
        if not_tensor:
            im /= 255  # 像素归一化到 [0, 1]
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """使用指定模型和参数对图像进行推理。"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """
        在推理前对输入图像执行 LetterBox 等几何预处理。

        参数:
            im (list[np.ndarray]): 图像列表，每张形状为 (H, W, 3)。

        返回:
            (list[np.ndarray]): 预处理后的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]
    #------替换函数内容，增加 ROI 提取逻辑
    def postprocess(self, preds, img, orig_imgs):
        """
        1) NMS
        2) 将每张图片的预测构造成 Results 对象
        3) 提取装甲板 ROI（优先 OBB 旋转裁剪，退化为 xyxy 裁剪）
        4) 将 ROI 附加到每个 Results（r.armor_rois）并缓存到 self.armor_rois
        """
        # 1) NMS（保持与你当前流程一致）
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        results = []
        self.armor_rois.clear()  # 清空全局缓存

        # 名称表（可能在 AutoBackend 上）
        names = getattr(self.model, 'names', None)
        # 允许用 --armor-classes 自定义；否则用默认候选（按你现有工程习惯改）
        armor_name_candidates = set(getattr(self.args, 'armor_classes', ['armor', 'armor_plate', 'plate']))

        for i, det in enumerate(preds):
            # 取原图
            im0 = orig_imgs[i].copy() if isinstance(orig_imgs, list) else orig_imgs.copy()
            H, W = im0.shape[:2]

            # 2) scale 回原图坐标（先对常规 xyxy）
            if len(det):
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            # 3) 组装 Results（先放常规 boxes/probs；masks 视任务而定）
            r = Results(
                path=None,
                boxes=det[:, :6] if len(det) else det,  # 兼容 boxes（xyxy, conf, cls）
                masks=None,
                probs=None,
                names=names,
                orig_img=im0
            )

            # 4) ROI 提取（优先 OBB；若无 OBB 则用 xyxy）
            rois = []

            # 4.1 先尝试从 r 或 det 中找到 OBB 信息（不同分支字段名不完全一致）
            # 常见位置：r.obb / r.boxes.rboxes / det.obb / det.rboxes / 自定义 attr
            obb_array = None
            # 从 r.boxes 下游结构尝试
            if hasattr(r, 'boxes') and r.boxes is not None:
                # 一些实现把 OBB 放在 r.boxes.rboxes 或 r.boxes.obb
                for key in ('rboxes', 'obb', 'xywhr', 'xywht'):
                    if hasattr(r.boxes, key):
                        obb_array = getattr(r.boxes, key, None)
                        if obb_array is not None:
                            try:
                                obb_array = np.asarray(obb_array, dtype=np.float32)
                            except Exception:
                                obb_array = None
                        if obb_array is not None:
                            break

            # 4.2 遍历每个检测，匹配装甲板类别
            if len(det):
                for j in range(det.shape[0]):
                    x1, y1, x2, y2, conf, cls = det[j].tolist()
                    cls = int(cls)
                    cls_name = (names[cls] if names and cls in range(len(names)) else str(cls))

                    # 若指定了装甲板类别名才提取 ROI
                    if cls_name not in armor_name_candidates:
                        continue

                    # 优先用 OBB 的第 j 个框
                    roi = None
                    if obb_array is not None and j < len(obb_array):
                        # 兼容若 obb 为 [cx,cy,w,h,theta] 或 [x,y,w,h,theta]；假定为像素坐标 + 弧度
                        cx, cy, w, h, theta = obb_array[j][:5]
                        # 容错：若是归一化，放大回像素
                        if max(cx, cy, w, h) <= 1.5:  # 粗略判断
                            cx, cy, w, h = cx * W, cy * H, w * W, h * H
                        quad = self._rbox_to_quad(cx, cy, w, h, theta)
                        try:
                            roi = self._crop_quad(im0, quad, out_size=(64, 64))
                        except Exception:
                            roi = None

                    # 回退：若没有 OBB 或裁剪失败，则用轴对齐裁剪
                    if roi is None:
                        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                        xi2, yi2 = min(W, int(x2)), min(H, int(y2))
                        if (xi2 - xi1) > 1 and (yi2 - yi1) > 1:
                            roi = im0[yi1:yi2, xi1:xi2].copy()

                    if roi is not None and roi.size > 0:
                        rois.append(roi)

            # 把 ROI 放进单张结果；同时缓存到 predictor 上（供外部读取）
            r.armor_rois = rois
            results.append(r)
            self.armor_rois.append(rois)

        return results
    def setup_source(self, source):
        """
        设置输入源及推理模式。

        参数:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor):
                推理输入源，可以是单张图像、文件夹、视频路径、流媒体 URL 或张量。
        """
        # 检查输入图像尺寸是否符合模型要求
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        # 根据输入源创建数据加载器
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "ch", 3),
        )
        # 记录输入源类型
        self.source_type = self.dataset.source_type

        # 判断是否为长序列(例如视频或大型数据集)
        long_sequence = (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # 超过 1000 张图像
            or any(getattr(self.dataset, "video_flag", [False]))  # 是否为视频流
        )
        if long_sequence:
            import torchvision  # 延迟导入，触发 torchvision 的 NMS 实现

            if not getattr(self, "stream", True):  # 若不是流式模式则发出警告
                LOGGER.warning(STREAM_WARNING)
        # 初始化视频写入器
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        在流媒体或视频上执行实时推理，并将结果保存到文件。

        参数:
            source: 输入源(图像、视频、流等)。
            model: 要加载或使用的模型。
            *args, **kwargs: 额外推理参数。

        返回:
            (generator): 逐帧输出的推理结果对象。
        """
        if self.args.verbose:
            LOGGER.info("")

        # 模型初始化
        if not self.model:
            self.setup_model(model)

        # 使用线程锁以保证多线程环境下推理安全
        with self._lock:
            # 每次调用 predict 时重新设置输入源
            self.setup_source(source if source is not None else self.args.source)

            # 若开启保存选项，则创建结果输出文件夹
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 模型预热(提高第一次推理速度)
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, self.model.ch, *self.imgsz)
                )
                self.done_warmup = True

            # 初始化统计变量
            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),  # 预处理计时
                ops.Profile(device=self.device),  # 推理计时
                ops.Profile(device=self.device),  # 后处理计时
            )

            # 执行推理起始回调
            self.run_callbacks("on_predict_start")

            # 遍历输入数据集(逐批推理)
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # --------- 图像预处理阶段 ---------
                with profilers[0]:
                    im = self.preprocess(im0s)

                # --------- 模型推理阶段 ---------
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        # 如果是特征嵌入任务，则直接输出张量结果
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds
                        continue

                # ---------后处理阶段 ---------
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # --------- 结果处理与保存 ---------
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        # 控制台打印、保存或显示推理结果
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # 批次结果打印
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results  # 将结果返回给上层调用者

        # ---------  推理完成后释放资源 ---------
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # 若开启窗口显示，则关闭所有窗口
        if self.args.show:
            cv2.destroyAllWindows()

        # ---------  最终结果信息打印 ---------
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}" % t
            )

        # 保存标签与文件路径信息
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} 个标签文件已保存至 {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"结果已保存到 {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """
        初始化 YOLO 模型，并将其设置为推理(评估)模式。

        参数:
            model (str | Path | torch.nn.Module, 可选): 要加载或使用的模型。
            verbose (bool): 是否打印详细信息。
        """
        self.model = AutoBackend(
            model=model or self.args.model,  # 加载模型文件或传入的模型对象
            device=select_device(self.args.device, verbose=verbose),  # 自动选择设备
            dnn=self.args.dnn,       # 是否使用 OpenCV DNN 模式
            data=self.args.data,     # 数据配置
            fp16=self.args.half,     # 半精度模式
            fuse=True,               # 模型融合优化
            verbose=verbose,
        )

        # 更新设备与精度设置
        self.device = self.model.device
        self.args.half = self.model.fp16

        # 若模型中保存有图像尺寸元数据，则沿用
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz

        # 切换为推理模式
        self.model.eval()

        # 尝试编译模型(若设备支持)
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """
        将推理结果写入文件或目录。

        参数:
            i (int): 当前批次中图像的索引。
            p (Path): 当前图像的路径。
            im (torch.Tensor): 预处理后的图像张量。
            s (list[str]): 批次状态信息字符串列表。

        返回:
            (str): 包含结果信息的字符串(用于控制台输出)。
        """
        string = ""  # 输出字符串初始化

        # 若图像缺少批次维度则添加
        if len(im.shape) == 3:
            im = im[None]

        # 判断输入源类型(视频流、单图像、张量输入等)
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            # 从状态字符串中提取帧号
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None

        # 生成文本结果路径(labels)
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])

        # 获取当前推理结果对象
        result = self.results[i]
        # === 两行对接：将 ROI 丢给数字分类器 ===
        if getattr(result, "armor_rois", None):
            digits, scores, _ = classify_rois(
                result.armor_rois,
                weights="digit_classifier.pt",
                device=self.device,
                conf_thr=0.6,  # 低于 0.6 的认为是“识别失败”
            )
            result.digits, result.digit_scores = digits, scores


        # ----------- 可视化部分 -----------
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

            # === 把数字叠到装甲板框上 ===
            if getattr(result, "digits", None) and getattr(result, "boxes", None) and len(result.boxes):
                import cv2
                armor_name_candidates = set(getattr(self.args, "armor_classes", ["armor", "armor_plate", "plate"]))
                names = getattr(self.model, "names", None)

                xyxy = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                k = 0  # digits 的游标，只给装甲板类框写数字
                for j in range(len(xyxy)):
                    cls_id = clss[j]
                    cls_name = (names[cls_id] if names and cls_id < len(names) else str(cls_id))
                    if cls_name not in armor_name_candidates:
                        continue
                    if k >= len(result.digits):
                        break

                    x1, y1, x2, y2 = map(int, xyxy[j])
                    txt = f"{result.digits[k]} ({result.digit_scores[k]:.2f})"
                    org = (x1, max(0, y1 - 6))
                    cv2.putText(self.plotted_img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    k += 1
            result.save_dir = self.save_dir.__str__()  # 为其他模块提供路径引用
            string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
                # ----------- 可视化部分 -----------
            if self.args.save or self.args.show:
                self.plotted_img = result.plot(
                    line_width=self.args.line_width,   # 绘制线条宽度
                    boxes=self.args.show_boxes,        # 是否显示边界框
                    conf=self.args.show_conf,          # 是否显示置信度
                    labels=self.args.show_labels,      # 是否显示标签
                    im_gpu=None if self.args.retina_masks else im[i],
                )

        # ----------- 结果保存部分 -----------
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """
        将视频或图像的推理结果保存为 mp4 或 jpg 文件。

        参数:
            save_path (Path): 结果保存路径。
            frame (int): 当前帧编号(仅视频模式下有效)。
        """
        im = self.plotted_img  # 绘制后的图像

        # ----------- 视频或流媒体模式 -----------
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # 单独存储帧图像

            # 初始化视频写入器
            if save_path not in self.vid_writer:
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                # 不同系统下的编码器选择
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 必须为整数，否则部分编码器报错
                    frameSize=(im.shape[1], im.shape[0]),  # (宽, 高)
                )

            # 写入视频帧
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # ----------- 静态图像模式 -----------
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # 保存为 JPG 格式(兼容性最好)

    def show(self, p: str = ""):
        """
        在窗口中显示图像。

        参数:
            p (str): 窗口标题或图像路径(用于标识)。
        """
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口缩放(Linux 专用)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (宽, 高)
        cv2.imshow(p, im)

        # 按下 q 键可退出(图像模式下延迟 300ms，视频流模式下 1ms)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):
            raise StopIteration

    def run_callbacks(self, event: str):
        """
        运行特定事件的所有回调函数。

        参数:
            event (str): 事件名称(如 "on_predict_start"、"on_predict_end")。
        """
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: callable):
        """
        为特定事件添加新的回调函数。

        参数:
            event (str): 事件名称。
            func (callable): 回调函数。
        """
        self.callbacks[event].append(func)
