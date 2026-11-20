
from pathlib import Path
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "helmet_detection_best.pt"
IMAGE_DIR = Path("helmet_yolo_dataset/images/test")
LABEL_DIR = Path("helmet_yolo_dataset/labels/test")

CONFIDENCE = 0.05
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640

# Existing dataset class -> new model class
CLASS_MAP = {
    0: 1,  # With Helmet -> helmet
    1: 2   # Without Helmet -> no helmet
}

model = YOLO(MODEL_PATH)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


stats = {
    0: {"TP": 0, "FP": 0, "FN": 0},
    1: {"TP": 0, "FP": 0, "FN": 0}
}

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

print("Model classes:", model.names)
print("Test images:", len(images))

for index, image_path in enumerate(images, 1):

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        print("Missing label:", label_path.name)
        continue

    ground_truth = {0: [], 1: []}

    for line in label_path.read_text().splitlines():
        values = line.strip().split()

        if len(values) != 5:
            continue

        cls, xc, yc, w, h = map(float, values)
        cls = int(cls)

        if cls not in CLASS_MAP:
            continue

        # YOLO normalized xywh -> pixel xyxy
        img = model.predictor
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size

        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height

        ground_truth[cls].append([x1, y1, x2, y2])

    result = model.predict(
        str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    predictions = {0: [], 1: []}

    for box in result.boxes:
        model_cls = int(box.cls[0].item())

        if model_cls not in [1, 2]:
            continue

        gt_cls = 0 if model_cls == 1 else 1

        predictions[gt_cls].append(
            box.xyxy[0].cpu().numpy().tolist()
        )

    for cls in [0, 1]:
        gt_boxes = ground_truth[cls]
        pred_boxes = predictions[cls]

        candidates = []

        for pi, pred in enumerate(pred_boxes):
            for gi, gt in enumerate(gt_boxes):
                iou = calculate_iou(pred, gt)

                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, pi, gi))

        candidates.sort(reverse=True)

        matched_preds = set()
        matched_gt = set()

        for iou, pi, gi in candidates:
            if pi not in matched_preds and gi not in matched_gt:
                matched_preds.add(pi)
                matched_gt.add(gi)

        stats[cls]["TP"] += len(matched_preds)
        stats[cls]["FP"] += len(pred_boxes) - len(matched_preds)
        stats[cls]["FN"] += len(gt_boxes) - len(matched_gt)

    if index % 10 == 0:
        print(f"Processed {index}/{len(images)} images")

print("\n========== EVALUATION RESULTS ==========")

for cls, name in [(0, "With Helmet"), (1, "Without Helmet")]:
    tp = stats[cls]["TP"]
    fp = stats[cls]["FP"]
    fn = stats[cls]["FN"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"\n{name}")
    print("TP:", tp, "| FP:", fp, "| FN:", fn)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

total_tp = sum(s["TP"] for s in stats.values())
total_fp = sum(s["FP"] for s in stats.values())
total_fn = sum(s["FN"] for s in stats.values())

precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0

print("\nOverall")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print("========================================")
