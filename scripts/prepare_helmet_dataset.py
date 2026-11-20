
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

# ---------------- CONFIGURATION ----------------

IMAGE_DIR = "helmet_dataset/images"
ANNOTATION_DIR = "helmet_dataset/annotations"
OUTPUT_DIR = "helmet_yolo_dataset"

CLASSES = {
    "With Helmet": 0,
    "Without Helmet": 1
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

# -----------------------------------------------

random.seed(RANDOM_SEED)

# Remove previous generated dataset only
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

valid_samples = []
excluded_samples = []
class_counts = Counter()

xml_files = [
    f for f in os.listdir(ANNOTATION_DIR)
    if f.lower().endswith(".xml")
]

print("Scanning annotations...")

for xml_file in xml_files:
    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        excluded_samples.append((xml_file, "Invalid XML"))
        continue

    image_name = root.findtext("filename")

    if not image_name:
        excluded_samples.append((xml_file, "Missing filename"))
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.isfile(image_path):
        excluded_samples.append((xml_file, "Missing image"))
        continue

    size = root.find("size")

    if size is None:
        excluded_samples.append((xml_file, "Missing image dimensions"))
        continue

    try:
        width = float(size.findtext("width"))
        height = float(size.findtext("height"))
    except (TypeError, ValueError):
        excluded_samples.append((xml_file, "Invalid dimensions"))
        continue

    if width <= 0 or height <= 0:
        excluded_samples.append((xml_file, "Invalid dimensions"))
        continue

    labels = []
    image_has_invalid_box = False

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name not in CLASSES:
            image_has_invalid_box = True
            break

        box = obj.find("bndbox")

        if box is None:
            image_has_invalid_box = True
            break

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            image_has_invalid_box = True
            break

        # Do not repair or scale suspicious coordinates.
        if not (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        ):
            image_has_invalid_box = True
            break

        # Convert Pascal VOC coordinates to YOLO format.
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASSES[class_name]

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    # Exclude the entire image if even one annotation is suspicious.
    if image_has_invalid_box or not labels:
        excluded_samples.append((xml_file, "Suspicious or empty annotations"))
        continue

    valid_samples.append({
        "image": image_path,
        "filename": image_name,
        "labels": labels
    })

# ---------------- DATASET SPLIT ----------------

random.shuffle(valid_samples)

total = len(valid_samples)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": valid_samples[:train_end],
    "val": valid_samples[train_end:val_end],
    "test": valid_samples[val_end:]
}

# ---------------- SAVE DATASET ----------------

final_counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter()
}

for split, samples in splits.items():
    for sample in samples:
        image_name = sample["filename"]
        image_stem = os.path.splitext(image_name)[0]

        destination_image = os.path.join(
            OUTPUT_DIR, "images", split, image_name
        )

        destination_label = os.path.join(
            OUTPUT_DIR, "labels", split, image_stem + ".txt"
        )

        shutil.copy2(sample["image"], destination_image)

        with open(destination_label, "w", encoding="utf-8") as f:
            f.write("\n".join(sample["labels"]))

        for label in sample["labels"]:
            class_id = int(label.split()[0])

            if class_id == 0:
                final_counts[split]["With Helmet"] += 1
            else:
                final_counts[split]["Without Helmet"] += 1

# ---------------- CREATE YAML ----------------

yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR).replace(os.sep, "/")}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: With Helmet
  1: Without Helmet
"""

yaml_path = os.path.join(OUTPUT_DIR, "helmet_data.yaml")

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

# ---------------- REPORT ----------------

with open(os.path.join(OUTPUT_DIR, "dataset_report.txt"), "w", encoding="utf-8") as f:
    f.write("HELMET DATASET REPORT\n")
    f.write("=====================\n\n")
    f.write(f"Original XML files: {len(xml_files)}\n")
    f.write(f"Usable images: {total}\n")
    f.write(f"Excluded images: {len(excluded_samples)}\n\n")

    for split, samples in splits.items():
        f.write(f"{split.upper()} images: {len(samples)}\n")
        f.write(f"With Helmet: {final_counts[split]['With Helmet']}\n")
        f.write(f"Without Helmet: {final_counts[split]['Without Helmet']}\n\n")

    f.write("EXCLUDED FILES:\n")
    for filename, reason in excluded_samples:
        f.write(f"{filename}: {reason}\n")

print("\n========== DATASET PREPARATION COMPLETE ==========")
print("Original XML files:", len(xml_files))
print("Usable images:", total)
print("Excluded images:", len(excluded_samples))

for split, samples in splits.items():
    print(f"\n{split.upper()}:")
    print("Images:", len(samples))
    print("With Helmet:", final_counts[split]["With Helmet"])
    print("Without Helmet:", final_counts[split]["Without Helmet"])

print("\nDataset YAML:", yaml_path)
print("Detailed report:", os.path.join(OUTPUT_DIR, "dataset_report.txt"))
