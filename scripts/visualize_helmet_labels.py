
import cv2
from pathlib import Path

images_dir = Path("helmet_yolo_dataset/images/train")
labels_dir = Path("helmet_yolo_dataset/labels/train")
output_dir = Path("annotation_review")

output_dir.mkdir(exist_ok=True)

image_names = [
    "BikesHelmets2",
    "BikesHelmets77",
    "BikesHelmets702"
]

classes = {
    0: "With Helmet",
    1: "Without Helmet"
}

for name in image_names:

    image_path = None

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / f"{name}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    label_path = labels_dir / f"{name}.txt"

    if image_path is None or not label_path.exists():
        print(f"Missing image or label: {name}")
        continue

    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        values = line.strip().split()

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(
            float, values[1:]
        )

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        label = classes.get(class_id, "Unknown")
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            image, label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )

    output_path = output_dir / f"{name}_review.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print("\nAnnotation visualization completed!")
