from ultralytics import YOLO
import cv2, os

model = YOLO("yolo11n-obb.pt")

# 使用绝对路径
img_path = "/home/lin/Desktop/deep_learning/ultralytics-obb/ultralytics/assets/bus.jpg"

results = model(img_path)

print("Detected armor ROI count:", len(model.predictor.armor_rois))

save_dir = "roi_test_output"
os.makedirs(save_dir, exist_ok=True)

for i, roi in enumerate(model.predictor.armor_rois):
    path = os.path.join(save_dir, f"roi_{i}.jpg")
    cv2.imwrite(path, roi)
    print(f"Saved ROI: {path}, shape={roi.shape}")
