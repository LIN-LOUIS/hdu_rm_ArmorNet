from pathlib import Path
from ultralytics import YOLO

# 1) 模型（你的 OBB 权重）
model = YOLO("yolo11n-obb.pt")  # 替换为你的自训权重也行

# 2) 测试图像路径（改成你的绝对路径最稳）
IMG = Path("/home/lin/Desktop/deep_learning/images/frame_00000.jpg")
assert IMG.exists(), f"Test image not found: {IMG}"

# 3) 推理（会自动在图上叠数字并保存到 runs/predict-* 目录）
res = model(str(IMG), task="obb", conf=0.25)
print("Images:", len(res))
r0 = res[0]
print("Armor ROI count:", len(getattr(r0, "armor_rois", []) or []))
print("Digits:", getattr(r0, "digits", None))
print("Scores:", getattr(r0, "digit_scores", None))
print("Done. Check the latest runs/predict* folder for output image.")
