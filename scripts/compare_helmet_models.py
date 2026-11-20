
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# LOAD ALL THREE MODELS
# --------------------------------------------------

models = {
    "Original": YOLO("helmet.pt"),
    "Fine-tuned": YOLO("helmet_finetuned.pt"),
    "New YOLO11n": YOLO("helmet_detection_best.pt")
}

# --------------------------------------------------
# TEST IMAGES
# --------------------------------------------------

test_folder = Path("helmet_yolo_dataset/images/test")

image_names = [
    "BikesHelmets604",
    "BikesHelmets700",
    "BikesHelmets526",
    "BikesHelmets628",
    "BikesHelmets440",
]

output_folder = Path("helmet_comparison")
output_folder.mkdir(exist_ok=True)

extensions = [".jpg", ".jpeg", ".png", ".webp"]

CONFIDENCE = 0.25

# --------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------

for name in image_names:

    image_path = None

    for ext in extensions:
        candidate = test_folder / f"{name}{ext}"

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"Image not found: {name}")
        continue

    original_image = cv2.imread(str(image_path))

    if original_image is None:
        print(f"Could not read: {image_path}")
        continue

    comparison_images = []

    print(f"\nImage: {name}")

    for model_name, model in models.items():

        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        annotated_image = result.plot()

        detection_count = len(result.boxes)

        print(
            f"{model_name}: {detection_count} detections"
        )

        cv2.putText(
            annotated_image,
            model_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        comparison_images.append(annotated_image)

    # Resize all images to the same dimensions
    target_height = min(
        img.shape[0] for img in comparison_images
    )

    resized_images = []

    for img in comparison_images:

        scale = target_height / img.shape[0]

        target_width = int(img.shape[1] * scale)

        resized = cv2.resize(
            img,
            (target_width, target_height)
        )

        resized_images.append(resized)

    # Side-by-side comparison
    comparison = np.hstack(resized_images)

    output_path = output_folder / f"{name}_comparison.jpg"

    cv2.imwrite(str(output_path), comparison)

    print(f"Saved: {output_path}")

print("\nThree-model comparison completed!")
