
from ultralytics import YOLO

MODEL_PATH = "helmet.pt"
DATA_YAML = "helmet_yolo_dataset/helmet_data.yaml"

model = YOLO(MODEL_PATH)

results = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    plots=True,
    workers=0
)

print("\n========== BASELINE RESULTS ==========")
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("Per-class mAP50:", results.box.maps)

print("\nBaseline evaluation completed.")
