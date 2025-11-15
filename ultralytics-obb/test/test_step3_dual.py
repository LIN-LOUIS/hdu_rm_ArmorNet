from ultralytics.models.yolo.obb.predict_dual import DualOBBPipeline
from pathlib import Path

MODEL = "yolo11n-obb.pt"  # 或训练权重
ARMOR_IMG = "/home/lin/Desktop/deep_learning/ultralytics-obb/ultralytics/assets/bus.jpg"
OUT_DIR = "runs/dual_step3"

pipeline = DualOBBPipeline(
    model_path=MODEL,
    digit_weights="digit_classifier.pt",  # 没有就先随便填，predict_dual 里会照样跑，只是数字随机
    device=None,
    out_dir=OUT_DIR,
    max_workers=2,
    conf=0.25,
)

pipeline.run(source=ARMOR_IMG, save=True, show=False)
print(f"[Step3] Done. 打开 {OUT_DIR}/ 下的 frame_000000.jpg 看是否有框 + 数字。")
