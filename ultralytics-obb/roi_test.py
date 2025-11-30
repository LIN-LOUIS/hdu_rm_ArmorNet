# ultralytics-obb/test/test_roi_collect.py
from ultralytics import YOLO
import cv2, os

model = YOLO("yolo11n-obb.pt")
img_dir = "armor_frames_dir"  # 放多帧装甲板截图的文件夹
save_dir = "digits_raw_roi"
os.makedirs(save_dir, exist_ok=True)

for name in os.listdir(img_dir):
    path = os.path.join(img_dir, name)
    res = model(path, task="obb", conf=0.25)
    rois = model.predictor.armor_rois
    print(path, "ROI num:", len(rois))
    for i, roi in enumerate(rois):
        out = os.path.join(save_dir, f"{os.path.splitext(name)[0]}_roi_{i}.jpg")
        cv2.imwrite(out, roi)
