# ultralytics-obb/ultralytics/models/yolo/obb/predict_dual.py
from __future__ import annotations
from ultralytics.models.digit_classifier import classify_rois

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch

from ultralytics import YOLO


class DualOBBPipeline:
    """
    A方案：YOLO-OBB 低阈值召回 -> ROI -> digit -> 过滤 + 融合分数 -> 可视化/保存
    依赖：predictor.py 已经往 result 填充 result.armor_rois (list[np.ndarray])
    """

    def __init__(
        self,
        model_path: str,
        digit_weights: Optional[str] = None,
        device: Optional[str] = None,
        out_dir: str = "runs/dual",
        max_workers: int = 2,
        conf: float = 0.05,
        digit_thres: float = 0.60,
        fuse_score: bool = True,
    ):
        self.yolo = YOLO(model_path)
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.conf = float(conf)

        # A方案关键参数
        self.digit_thres = float(digit_thres)
        self.fuse_score = bool(fuse_score)

        #digit 模型只加载一次（不要每帧重新加载）
        # digit 使用函数式接口（与你现有 digit_classifier.py 一致）
        self.digit_weights = digit_weights
        self.device = device

    def _classify(self, rois):
        """
        rois: List[np.ndarray] (BGR)
        return: digits, probs
        """
        if rois is None or len(rois) == 0:
            return [], []
        digits, probs, _ = classify_rois(rois, self.digit_weights, self.device)
        return digits, probs


    # ----------------- 绘制（用 OBB 多边形定位） -----------------
    @staticmethod
    def _draw(result, digits: List[int], probs: List[float], digit_thres: float, fuse: bool):
        """
        在 result.plot() 的基础上叠加 digit。用 result.obb.xyxyxyxy 定位文字更准。
        """
        im = result.plot()
        if getattr(result, "obb", None) is None or len(result.obb) == 0:
            return im

        polys = result.obb.xyxyxyxy.detach().cpu().numpy()  # (N, 8)
        det_confs = result.obb.conf.detach().cpu().numpy()  # (N,)

        n = min(len(polys), len(digits), len(probs))
        for i in range(n):
            d = int(digits[i])
            p = float(probs[i])
            dc = float(det_confs[i])

            #   A方案：digit 过滤假框
            if d < 0 or p < digit_thres:
                continue

            final = (dc * p) if fuse else dc

            pts = polys[i].reshape(4, 2).astype(np.int32)
            x, y = pts[0]
            txt = f"{d} det={dc:.2f} dig={p:.2f} f={final:.2f}"

            cv2.putText(
                im,
                txt,
                (int(x), max(0, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return im

    # ----------------- 主入口 -----------------
    def run(self, source: str | int, save: bool = True, show: bool = False):
        results_gen = self.yolo(source, task="obb", conf=self.conf, stream=True)

        frame_idx = 0
        img_dir = self.out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        #   GPU 推理 + 多线程有时反而更慢/不稳定；这里保留线程池但建议 workers=1~2
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending = []

            for result in results_gen:
                rois = getattr(result, "armor_rois", None) or []

                # 提交 digit 任务（此时不会重复加载模型，因为模型已在 __init__ 里加载）
                fut = executor.submit(self._classify, rois)
                pending.append((frame_idx, result, fut))

                # 控制队列长度
                if len(pending) >= self.max_workers * 3:
                    idx0, res0, fut0 = pending.pop(0)
                    digits0, probs0 = fut0.result()

                    im0 = self._draw(res0, digits0, probs0, self.digit_thres, self.fuse_score)

                    if save:
                        cv2.imwrite(str(img_dir / f"frame_{idx0:06d}.jpg"), im0)
                    if show:
                        cv2.imshow("dual", im0)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_idx += 1

            # flush
            for idx0, res0, fut0 in pending:
                digits0, probs0 = fut0.result()
                im0 = self._draw(res0, digits0, probs0, self.digit_thres, self.fuse_score)
                if save:
                    cv2.imwrite(str(img_dir / f"frame_{idx0:06d}.jpg"), im0)
                if show:
                    cv2.imshow("dual", im0)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if show:
            cv2.destroyAllWindows()

        print(f"[DualOBBPipeline] Done. Frames: {frame_idx}, saved to: {self.out_dir}")
