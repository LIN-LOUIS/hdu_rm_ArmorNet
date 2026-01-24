#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一入口脚本 main.py

支持模式:
  1) 检测 (detect):只跑 YOLO / YOLO-OBB 检测
  2) 并行推理 (dual):YOLO-OBB + 数字分类 两路并行推理
  3) 导出 (export):导出 OpenVINO 等格式

使用示例:

  # 1. 仅检测 OBB(单图/视频/摄像头)
  python main.py detect \
      --model yolo11n-obb.pt \
      --source ultralytics-obb/assets/armor_sample.jpg \
      --task obb \
      --conf 0.25

  # 2. OBB + 数字识别 并行推理
  python main.py dual \
      --model yolo11n-obb.pt \
      --digit-weights digit_classifier.pt \
      --source ultralytics-obb/assets/armor_sample.jpg \
      --conf 0.25

  # 3. 导出 OpenVINO
  python main.py export \
      --model yolo11n-obb.pt \
      --imgsz 1024 \
      --half
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


# ========= detect 模式 =========
def run_detect(args: argparse.Namespace):
    """
    只使用 YOLO / YOLO-OBB 做检测推理。
    """
    model = YOLO(args.model)

    # 根据 task 选择任务:obb / detect / segment / pose...
    task = args.task or "detect"
    conf = args.conf

    print(f"[main.detect] model={args.model}, task={task}, source={args.source}, conf={conf}")

    # 直接使用 Ultralytics 的高级接口
    results = model(
        args.source,
        task=task,
        conf=conf,
        save=not args.nosave,
        show=args.show,
    )

    # 打印一下简单信息
    print(f"[main.detect] Done. Images: {len(results)}")
    if not args.nosave:
        print("[main.detect] 结果默认保存在 runs/predict* 目录下。")


# ========= dual 模式 =========
def run_dual(args: argparse.Namespace):
    """
    YOLO-OBB + 数字分类并行推理。
    依赖:ultralytics.models.yolo.obb.predict_dual.DualOBBPipeline
    """
    from ultralytics.models.yolo.obb.predict_dual import DualOBBPipeline

    print(
        f"[main.dual] model={args.model}, "
        f"digit_weights={args.digit_weights}, source={args.source}, conf={args.conf}"
    )

    pipeline = DualOBBPipeline(
        model_path=args.model,
        digit_weights=args.digit_weights,
        device=args.device,
        out_dir=args.project,
        max_workers=args.workers,
        conf=args.conf,
        digit_thres=args.digit_thres,
        fuse_score=args.fuse,
    )

    pipeline.run(
        source=args.source,
        save=not args.nosave,
        show=args.show,
    )


# ========= export 模式 =========
def run_export(args: argparse.Namespace):
    """
    使用 Ultralytics 的导出接口，导出不同格式(例如 OpenVINO)。
    典型用法:导出 OpenVINO 的 .xml / .bin

      python main.py export --model yolo11n-obb.pt --format openvino --imgsz 1024
    """
    model = YOLO(args.model)

    export_args = {
        "format": args.format,          # 导出格式，如 "openvino", "onnx", "engine", ...
        "imgsz": args.imgsz,           # 输入尺寸
        "half": args.half,             # 是否半精度
        "dynamic": args.dynamic,       # 是否动态图
        "simplify": args.simplify,     # 是否简化(对 onnx 有用)
        "int8": args.int8,             # 是否 INT8 量化(部分后端支持)
        "device": args.device,         # 导出设备
    }

    print(f"[main.export] model={args.model}, args={export_args}")

    out_path = model.export(**export_args)
    print(f"[main.export] 导出完成: {out_path}")


# ========= 参数解析 =========
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified entry for YOLO-OBB + Digit pipeline")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----- detect 子命令 -----
    p_detect = subparsers.add_parser("detect", help="仅 YOLO / YOLO-OBB 检测推理")
    p_detect.add_argument("--model", type=str, required=True, help="模型权重 .pt 路径")
    p_detect.add_argument("--source", type=str, required=True, help="输入源(图像/视频/摄像头等)")
    p_detect.add_argument("--task", type=str, default="obb", help="任务类型:obb / detect / segment / pose 等")
    p_detect.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p_detect.add_argument("--nosave", action="store_true", help="不保存结果，只显示/打印")
    p_detect.add_argument("--show", action="store_true", help="是否弹窗显示")
    p_detect.set_defaults(func=run_detect)

    # ----- dual 子命令 -----
    p_dual = subparsers.add_parser("dual", help="YOLO-OBB + 数字分类 并行推理")
    p_dual.add_argument("--model", type=str, required=True, help="YOLO11-OBB 模型权重 .pt")
    p_dual.add_argument("--digit-weights", type=str, default=None, help="digit_classifier.pt 权重路径")
    p_dual.add_argument("--source", type=str, required=True, help="输入源(图像/视频/摄像头/视频流等)")
    p_dual.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p_dual.add_argument("--device", type=str, default=None, help="推理设备，如 cuda:0 / cpu")
    p_dual.add_argument("--project", type=str, default="runs/dual_main", help="输出目录")
    p_dual.add_argument("--workers", type=int, default=2, help="ThreadPoolExecutor worker 数")
    p_dual.add_argument("--nosave", action="store_true", help="不保存结果，只显示/调试")
    p_dual.add_argument("--show", action="store_true", help="是否弹窗显示")
    p_dual.set_defaults(func=run_dual)
    p_dual.add_argument("--digit-thres", type=float, default=0.60, help="数字置信度阈值，低于则丢框")
    p_dual.add_argument("--fuse", action="store_true", help="融合分数 final=det_conf*digit_prob")

    # ----- export 子命令 -----
    p_export = subparsers.add_parser("export", help="导出不同部署格式(OpenVINO / ONNX / TensorRT 等)")
    p_export.add_argument("--model", type=str, required=True, help="模型权重 .pt 路径")
    p_export.add_argument("--format", type=str, default="openvino", help="导出格式，如 openvino / onnx / engine / torchscript 等")
    p_export.add_argument("--imgsz", type=int, default=1024, help="导出时的输入尺寸(单值代表正方形)")
    p_export.add_argument("--half", action="store_true", help="导出为 FP16 半精度(支持的后端)")
    p_export.add_argument("--dynamic", action="store_true", help="是否启用动态输入尺寸")
    p_export.add_argument("--simplify", action="store_true", help="是否简化导出图(对 onnx 有用)")
    p_export.add_argument("--int8", action="store_true", help="是否 INT8 量化(部分后端支持)")
    p_export.add_argument("--device", type=str, default=None, help="导出设备，如 cuda:0 / cpu")
    p_export.set_defaults(func=run_export)

    return parser.parse_args()


def main():
    args = get_args()

    # 做个简单的路径检查
    if hasattr(args, "model"):
        m = Path(args.model)
        if not m.exists():
            print(f"[main] 警告:模型路径不存在:{m}(若使用 hub/远程模型可忽略)")

    # 调用对应子命令处理函数
    args.func(args)


if __name__ == "__main__":
    main()
