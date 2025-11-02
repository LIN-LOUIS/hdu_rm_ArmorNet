# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """
    扩展自 DetectionPredictor 的旋转边界框（OBB）预测类。

    该预测器专用于处理旋转目标检测任务，可对输入图像进行推理，
    输出带有旋转角度的目标检测结果。

    属性：
        args (namespace): 预测器的配置参数。
        model (torch.nn.Module): 已加载的 YOLO-OBB 模型。

    示例：
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb import OBBPredictor
        >>> args = dict(model="yolo11n-obb.pt", source=ASSETS)
        >>> predictor = OBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 OBBPredictor，并允许传入模型或数据配置的自定义覆盖参数。

        参数：
            cfg (dict, 可选): 预测器的默认配置。
            overrides (dict, 可选): 自定义配置项，会覆盖默认配置。
            _callbacks (list, 可选): 在预测过程中触发的回调函数列表。

        示例：
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.obb import OBBPredictor
            >>> args = dict(model="yolo11n-obb.pt", source=ASSETS)
            >>> predictor = OBBPredictor(overrides=args)
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"  # 将任务类型设置为旋转框检测（Oriented Bounding Box）

    def construct_result(self, pred, img, orig_img, img_path):
        """
        根据模型预测结果构建结果对象（Results）。

        参数：
            pred (torch.Tensor): 模型的预测输出，形状为 (N, 7)，
                含 [x, y, w, h, confidence, class_id, angle]。
            img (torch.Tensor): 预处理后的图像，形状为 (B, C, H, W)。
            orig_img (np.ndarray): 原始输入图像（未经过预处理）。
            img_path (str): 原始图像的路径。

        返回：
            (Results): 包含原图、路径、类别名称以及旋转边界框（OBB）的结果对象。
        """
        # 将预测框中的旋转角度进行标准化处理
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        # 将预测框从推理尺寸缩放回原始图像尺寸
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        # 拼接置信度与类别信息
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        # 构建结果对象
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)
