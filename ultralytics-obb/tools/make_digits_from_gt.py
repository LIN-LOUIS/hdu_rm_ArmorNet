import os
from pathlib import Path
import cv2
import numpy as np
import yaml

# 你的类别名（从 yaml 里读），把它映射到“数字标签”
# 例：1_blue / 1_red -> 1；Sen -> None（跳过）
def build_name_to_digit(names_dict):
    name_to_digit = {}
    for k, v in names_dict.items():
        name = str(v)
        if name.lower() == "sen":
            name_to_digit[int(k)] = None
            continue
        # 支持 "1_blue" / "1_red" / "5" / "0"
        if "_" in name:
            d = name.split("_")[0]
        else:
            d = name
        if d.isdigit():
            name_to_digit[int(k)] = int(d)
        else:
            name_to_digit[int(k)] = None
    return name_to_digit

def read_names_from_yaml(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names", {})
    # names 可能是 dict 或 list
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names

def crop_quad_to_64(im_bgr, quad_xy, out_size=64, scale=1.20):
    # quad_xy: (4,2) 像素坐标
    quad = quad_xy.astype(np.float32)

    # --- 1) 先把点顺序整理成 tl,tr,br,bl ---
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).reshape(-1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    quad2 = np.array([tl, tr, br, bl], dtype=np.float32)

    # --- 2) 以中心点为基准，整体放大一点（给上下左右留边）---
    c = quad2.mean(axis=0, keepdims=True)          # (1,2)
    quad2 = (quad2 - c) * scale + c                # 放大

    # --- 3) 边界裁剪，避免越界 ---
    H, W = im_bgr.shape[:2]
    quad2[:, 0] = np.clip(quad2[:, 0], 0, W - 1)
    quad2[:, 1] = np.clip(quad2[:, 1], 0, H - 1)

    # --- 4) 透视到 out_size x out_size ---
    dst = np.array([[0, 0],
                    [out_size - 1, 0],
                    [out_size - 1, out_size - 1],
                    [0, out_size - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad2, dst)
    roi = cv2.warpPerspective(im_bgr, M, (out_size, out_size))
    return roi


def main():
    # ====== 你要改的三个路径 ======
    data_yaml = "cfg/armor_obb.yaml"  
    dataset_root = Path("/home/lin/Desktop/deep_learning/armor_obb_dataset")
    out_root = Path("/home/lin/Desktop/deep_learning/digits_dataset")
    split = "train"  # 也可以 "val"

    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    out_root.mkdir(parents=True, exist_ok=True)

    names = read_names_from_yaml(data_yaml)
    name_to_digit = build_name_to_digit(names)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()

    saved = 0
    skipped = 0

    for img_path in img_paths:
        label_path = lab_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1
            continue
        H, W = im.shape[:2]

        lines = label_path.read_text().strip().splitlines()
        for li, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            digit = name_to_digit.get(cls, None)
            if digit is None:
                continue  # 跳过 Sen 等

            pts = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H

            roi = crop_quad_to_64(im, pts, out_size=64, scale=1.25)

            # 可选：简单过滤太黑/太空的
            if roi.mean() < 5:
                continue

            save_dir = out_root / str(digit)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{img_path.stem}_{li}.jpg"
            cv2.imwrite(str(save_path), roi)
            saved += 1

    print(f"[DONE] saved={saved}, skipped_images={skipped}")
    print(f"Output: {out_root}")

if __name__ == "__main__":
    main()
