import os
from pathlib import Path
import cv2
import numpy as np

# ====== 你自己的路径 ======
ROOT = Path("/home/lin/Desktop/deep_learning/armor_obb_dataset")
IMG_DIRS = [ROOT/"images/train", ROOT/"images/val"]
LBL_DIRS = [ROOT/"labels/train", ROOT/"labels/val"]

OUT = Path("/home/lin/Desktop/deep_learning/digits_dataset")
OUT.mkdir(parents=True, exist_ok=True)

# class_id -> digit, Sen(6) 跳过
CLS2DIG = {
    0: 0,
    1: 1, 7: 1,
    2: 2, 8: 2,
    3: 3, 9: 3,
    4: 4, 10: 4,
    5: 5,
    6: None,  # Sen
}

def warp_quad(im, quad, out_size=96):
    # quad: 4x2, 顺序任意时建议先做排序；这里假设你的标注点顺序稳定（一般是顺时针）
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(im, M, (out_size, out_size))

def yolo_obb_line_to_quad(line, W, H):
    parts = line.strip().split()
    if len(parts) != 9:
        return None, None
    cls = int(parts[0])
    pts = list(map(float, parts[1:]))
    quad = np.array([[pts[0]*W, pts[1]*H],
                     [pts[2]*W, pts[3]*H],
                     [pts[4]*W, pts[5]*H],
                     [pts[6]*W, pts[7]*H]], dtype=np.float32)
    return cls, quad

def iter_images(img_dir):
    exts = {".jpg",".jpeg",".png",".bmp"}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

def main():
    saved = 0
    skipped = 0

    for img_dir, lbl_dir in zip(IMG_DIRS, LBL_DIRS):
        for img_path in iter_images(img_dir):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            im = cv2.imread(str(img_path))
            if im is None:
                continue
            H, W = im.shape[:2]

            lines = lbl_path.read_text().strip().splitlines()
            for k, line in enumerate(lines):
                cls_id, quad = yolo_obb_line_to_quad(line, W, H)
                if cls_id is None:
                    continue

                digit = CLS2DIG.get(cls_id, None)
                if digit is None:  # Sen 或未知
                    skipped += 1
                    continue

                roi = warp_quad(im, quad, out_size=96)  # 先裁 96x96（比 64 更稳）
                if roi is None or roi.size == 0:
                    skipped += 1
                    continue

                out_dir = OUT / str(digit)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{img_path.stem}_{k}.jpg"
                cv2.imwrite(str(out_path), roi)
                saved += 1

    print(f"Done. saved={saved}, skipped={skipped}, out={OUT}")

if __name__ == "__main__":
    main()
