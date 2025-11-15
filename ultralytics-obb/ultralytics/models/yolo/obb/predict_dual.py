# Ultralytics 🚀 Dual OBB + Digit Classifier Pipeline
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.models.digit_classifier import classify_rois


class DualOBBPipeline:
    """
    YOLO-OBB + 数字分类 并行推理流水线

    流程：
        1. 使用 YOLO-OBB 对输入图像/视频进行检测，得到 Results（在 Step1已经加了 armor_rois）。
        2. 将每帧的 armor_rois 丢给数字分类器 classify_rois，在 ThreadPoolExecutor 中并行跑。
        3. 将分类结果（digit + score）写回当前帧结果，并在图像上叠加可视化数字。
        4. 保存到 out_dir，并可选窗口显示。

    注意：
        - 依赖：Step1 已经在 predictor 中填充 result.armor_rois（list[np.ndarray]）
        - 依赖：Step2 已经实现 ultralytics.models.digit_classifier.classify_rois
    """

    def __init__(
        self,
        model_path: str,
        digit_weights: Optional[str] = None,
        device: Optional[str] = None,
        out_dir: str = "runs/dual",
        max_workers: int = 2,
        conf: float = 0.25,
    ):
        """
        参数：
            model_path: YOLO11-OBB 权重路径 (.pt)
            digit_weights: 数字分类器权重路径（digit_classifier.pt），可为 None（则使用随机初始化，便于调试）
            device: 推理设备字符串（如 'cuda:0' 或 'cpu'），为空自动选择
            out_dir: 输出目录，保存叠加了数字的图片/帧
            max_workers: 线程池最大 worker 数（分类任务并行度）
            conf: YOLO 检测置信度阈值
        """
        self.yolo = YOLO(model_path)
        self.digit_weights = digit_weights
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.conf = conf

    # --------- 内部工具：把数字画到一帧上 ---------
    @staticmethod
    def _draw_digits_on_result(result, digits, scores):
        """
        在 result.plot() 返回的图像上叠加数字和置信度。

        假设：
            - result.boxes 与 armor_rois 顺序一一对应（Step1 中按装甲板顺序抽取 ROI）
        """
        # 基础可视化（只画框/标签）
        im = result.plot()
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return im

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        k = 0  # digits 游标
        for j in range(len(xyxy)):
            if k >= len(digits):
                break

            x1, y1, x2, y2 = map(int, xyxy[j])
            txt = str(digits[k])
            if scores:
                txt = f"{txt}({scores[k]:.2f})"

            org = (x1, max(0, y1 - 6))
            cv2.putText(
                im,
                txt,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            k += 1

        return im

    # --------- 主入口：并行推理 ---------
    def run(self, source: str | int, save: bool = True, show: bool = False):
        """
        执行并行推理。

        参数：
            source: 输入源，可以是图像路径/视频路径/摄像头编号等（与 YOLO 原生一致）
            save: 是否保存结果图像/视频帧到 out_dir
            show: 是否弹窗显示（按 q 退出）

        返回：
            无（结果保存在 out_dir，终端会打印帧数等信息）
        """
        # 使用 YOLO 原生流模式，逐帧拿到 Results
        results_gen = self.yolo(source, task="obb", conf=self.conf, stream=True)

        frame_idx = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 线程池：用来并行跑 classify_rois
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 当前正在进行中的 future 列表（避免无限堆积）
            pending = []

            for result in results_gen:
                rois = getattr(result, "armor_rois", None) or []

                # 1) 提交分类任务到线程池
                fut = executor.submit(
                    classify_rois,
                    rois,
                    self.digit_weights,
                    self.device,
                )
                pending.append((frame_idx, result, fut))

                # 限制 pending 的长度，防止视频很长时显存/内存堆积
                if len(pending) >= self.max_workers * 3:
                    idx0, res0, fut0 = pending.pop(0)
                    digits0, scores0, _ = fut0.result()
                    res0.digits, res0.digit_scores = digits0, scores0
                    im0 = self._draw_digits_on_result(res0, digits0, scores0)

                    if save:
                        save_path = self.out_dir / f"frame_{idx0:06d}.jpg"
                        cv2.imwrite(str(save_path), im0)
                    if show:
                        cv2.imshow("dual", im0)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx += 1

            # 2) 把剩余 pending 的帧全部取完
            for idx0, res0, fut0 in pending:
                digits0, scores0, _ = fut0.result()
                res0.digits, res0.digit_scores = digits0, scores0
                im0 = self._draw_digits_on_result(res0, digits0, scores0)

                if save:
                    save_path = self.out_dir / f"frame_{idx0:06d}.jpg"
                    cv2.imwrite(str(save_path), im0)
                if show:
                    cv2.imshow("dual", im0)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if show:
            cv2.destroyAllWindows()
        print(f"[DualOBBPipeline] Done. Frames: {frame_idx}, saved to: {self.out_dir}")


# --------- 命令行入口：方便你直接 python predict_dual.py 跑 ---------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("YOLO-OBB + Digit Classifier Dual Inference")
    parser.add_argument("--model", type=str, required=True, help="yolo11-obb .pt 权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入源（图像/视频/摄像头等）")
    parser.add_argument("--digit-weights", type=str, default=None, help="digit_classifier.pt 权重路径")
    parser.add_argument("--device", type=str, default=None, help="推理设备，如 cuda:0 / cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--project", type=str, default="runs/dual", help="输出目录")
    parser.add_argument("--workers", type=int, default=2, help="线程池并行 worker 数")
    parser.add_argument("--nosave", action="store_true", help="不保存结果，只显示/调试")
    parser.add_argument("--show", action="store_true", help="是否实时显示窗口")
    args = parser.parse_args()

    pipeline = DualOBBPipeline(
        model_path=args.model,
        digit_weights=args.digit_weights,
        device=args.device,
        out_dir=args.project,
        max_workers=args.workers,
        conf=args.conf,
    )
    pipeline.run(
        source=args.source,
        save=not args.nosave,
        show=args.show,
    )
