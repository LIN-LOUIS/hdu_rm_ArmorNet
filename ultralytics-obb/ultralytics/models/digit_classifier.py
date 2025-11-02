# Ultralytics 🚀 Digit Classifier (0-9) for Armor ROI
# 放置路径：ultralytics/models/digit_classifier.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) 轻量CNN骨干 - OpenVINO
# ----------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class DigitClassifier(nn.Module):
    """
    输入: Bx3x64x64 (RGB)
    输出: Bx10 (logits)
    结构: DW卷积堆叠 + GAP + FC
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),  # 64->32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage = nn.Sequential(
            DepthwiseSeparableConv(16, 32, 3, 1, 1),  # 32x32
            DepthwiseSeparableConv(32, 64, 3, 2, 1),  # 32->16
            DepthwiseSeparableConv(64, 96, 3, 1, 1),
            DepthwiseSeparableConv(96, 128, 3, 2, 1),  # 16->8
            DepthwiseSeparableConv(128, 160, 3, 1, 1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


# ----------------------------
# 2) 预处理 ,推理工具
# ----------------------------
def _preprocess_roi_bgr(roi_bgr: np.ndarray, size: int = 64) -> torch.Tensor:
    """
    输入: BGR ndarray(H,W,3)
    输出: torch.FloatTensor(3,64,64), 归一化到[0,1]
    """
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("Empty ROI for digit classifier.")
    # resize & pad 到正方形（保持比例，避免拉伸数字）
    h, w = roi_bgr.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=roi_bgr.dtype)
    y0 = (s - h) // 2
    x0 = (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = roi_bgr
    im = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # 3xHxW
    return t


@torch.inference_mode()
def classify_rois(
    rois: List[np.ndarray],
    weights: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    num_classes: int = 10,
) -> Tuple[List[int], List[float], np.ndarray]:
    """
    直接喂 ROI 列表进行分类（与你的 Step1 无缝衔接）
    返回: (pred_digits, pred_scores, logits_np)
    """
    if len(rois) == 0:
        return [], [], np.zeros((0, num_classes), dtype=np.float32)

    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DigitClassifier(num_classes=num_classes).to(dev).eval()
    if weights and Path(weights).exists():
        ckpt = torch.load(weights, map_location=dev)
        # 兼容 {'model':state_dict} / 直接state_dict 两种保存格式
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)

    # 预处理打包
    xs = []
    for roi in rois:
        try:
            xs.append(_preprocess_roi_bgr(roi))  # 3x64x64
        except Exception:
            continue
    if not xs:
        return [], [], np.zeros((0, num_classes), dtype=np.float32)

    X = torch.stack(xs, dim=0).to(dev)  # Bx3x64x64

    preds_digits: List[int] = []
    preds_scores: List[float] = []
    all_logits: List[np.ndarray] = []

    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        logits = model(X[start:end])  # BxC
        probs = F.softmax(logits, dim=1)
        conf, cls = probs.max(dim=1)
        preds_digits.extend(cls.tolist())
        preds_scores.extend(conf.tolist())
        all_logits.append(logits.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, num_classes), dtype=np.float32)
    return preds_digits, preds_scores, logits_np


# ----------------------------
# 3) 训练脚本（最小可用）
# 数据集组织：dataset_root/
#     ├── 0/*.png
#     ├── 1/*.png
#     └── ...
#     └── 9/*.png
# ----------------------------
class SimpleDigitFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, size: int = 64):
        self.root = Path(root)
        self.size = size
        self.samples: List[Tuple[Path, int]] = []
        for c in range(10):
            cls_dir = self.root / str(c)
            if cls_dir.is_dir():
                for p in cls_dir.rglob("*"):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        self.samples.append((p, c))
        if not self.samples:
            raise FileNotFoundError(f"No image files found in: {self.root} (expect subfolders 0-9)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        t = _preprocess_roi_bgr(im, size=self.size)  # 3x64x64
        return t, y


def train_digit_classifier(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weights_out: str = "digit_classifier.pt",
    device: Optional[str] = None,
):
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = SimpleDigitFolder(data_root, size=64)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = DigitClassifier(num_classes=10).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for X, y in dl:
            X = X.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                logits = model(X)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / max(1, total)
        epoch_loss = loss_sum / max(1, total)
        sched.step()
        print(f"[Epoch {ep:02d}] loss={epoch_loss:.4f} acc={acc:.4f}")

        # 简单保存最佳
        if acc > best_acc:
            best_acc = acc
            ckpt = {"model": model.state_dict(), "acc": acc, "epoch": ep}
            torch.save(ckpt, weights_out)
            print(f"  ↳ saved best to {weights_out} (acc={acc:.4f})")

    print(f"Training done. Best acc={best_acc:.4f}, weights={weights_out}")


# ----------------------------
# 4) ONNX 导出（后续可转 OpenVINO）
# ----------------------------
def export_onnx(weights_in: Optional[str], onnx_out: str = "digit_classifier.onnx", opset: int = 12):
    dev = torch.device("cpu")
    model = DigitClassifier(num_classes=10).to(dev).eval()

    if weights_in and Path(weights_in).exists():
        ckpt = torch.load(weights_in, map_location=dev)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)

    dummy = torch.randn(1, 3, 64, 64, device=dev)
    torch.onnx.export(
        model, dummy, onnx_out,
        input_names=["images"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX to: {onnx_out}")


# ----------------------------
# 5) 作为脚本使用
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["train", "export", "infer"])
    ap.add_argument("--data", type=str, help="dataset root for training (folders 0..9)")
    ap.add_argument("--weights", type=str, default=None, help="weights path for load/save")
    ap.add_argument("--onnx", type=str, default="digit_classifier.onnx")
    ap.add_argument("--img", type=str, default=None, help="single ROI image for infer test")
    args = ap.parse_args()

    if args.mode == "train":
        assert args.data, "--data is required for training"
        train_digit_classifier(args.data, weights_out=(args.weights or "digit_classifier.pt"))
    elif args.mode == "export":
        export_onnx(args.weights, onnx_out=args.onnx)
    elif args.mode == "infer":
        assert args.img, "--img is required for infer"
        im = cv2.imread(args.img, cv2.IMREAD_COLOR)
        digits, scores, _ = classify_rois([im], weights=args.weights)
        print("digit:", digits[0] if digits else None, "score:", scores[0] if scores else None)
