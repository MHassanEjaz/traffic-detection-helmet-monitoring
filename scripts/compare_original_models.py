
from ultralytics import YOLO

DATA_YAML = "helmet_yolo_dataset/helmet_data.yaml"

MODELS = [
    "helmet.pt",
    "helmet_finetuned.pt"
]

for model_path in MODELS:
    print("\n" + "=" * 55)
    print("Evaluating:", model_path)
    print("=" * 55)

    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False
    )

    print("\nOverall Results:")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    print("\nPer-class Results:")
    for class_id, class_name in model.names.items():
        print(
            f"{class_name}: "
            f"mAP50 = {metrics.box.maps[class_id]:.4f}"
        )

print("\nComparison completed.")
