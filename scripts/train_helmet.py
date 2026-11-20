
import os
import shutil
from ultralytics import YOLO

MODEL_PATH = "helmet.pt"
DATA_YAML = "helmet_yolo_dataset/helmet_data.yaml"

print("Loading existing helmet model...")
model = YOLO(MODEL_PATH)

print("\nStarting fine-tuning...")

results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    patience=8,
    optimizer="auto",
    plots=True,
    project="runs/helmet",
    name="finetune",
    exist_ok=False,
    seed=42,
    deterministic=True
)

best_model = os.path.join(
    "runs", "helmet", "finetune", "weights", "best.pt"
)

if not os.path.exists(best_model):
    raise FileNotFoundError(
        "Training completed but best.pt was not found."
    )

shutil.copy2(best_model, "helmet_finetuned.pt")

print("\n========== TRAINING COMPLETE ==========")
print("Best model:", best_model)
print("Copied model: helmet_finetuned.pt")
