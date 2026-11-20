
import os
import xml.etree.ElementTree as ET
from collections import Counter

IMAGE_DIR = "helmet_dataset/images"
ANNOTATION_DIR = "helmet_dataset/annotations"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

image_files = {
    f.lower() for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
}

xml_files = [
    f for f in os.listdir(ANNOTATION_DIR)
    if f.lower().endswith(".xml")
]

missing_images = []
invalid_boxes = []
missing_dimensions = []
class_counts = Counter()

for xml_file in xml_files:
    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    try:
        root = ET.parse(xml_path).getroot()

        filename = root.findtext("filename")
        if not filename or filename.lower() not in image_files:
            missing_images.append(filename or xml_file)

        size = root.find("size")

        if size is None:
            missing_dimensions.append(xml_file)
            continue

        width = int(size.findtext("width", 0))
        height = int(size.findtext("height", 0))

        for obj in root.findall("object"):
            name = obj.findtext("name", "Unknown").strip()
            class_counts[name] += 1

            box = obj.find("bndbox")
            if box is None:
                invalid_boxes.append(xml_file)
                continue

            xmin = int(box.findtext("xmin", 0))
            ymin = int(box.findtext("ymin", 0))
            xmax = int(box.findtext("xmax", 0))
            ymax = int(box.findtext("ymax", 0))

            if not (
                0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            ):
                invalid_boxes.append(xml_file)

    except Exception as e:
        print("Error:", xml_file, e)

print("\n========== DATASET AUDIT ==========")
print("Total images:", len(image_files))
print("Total XML files:", len(xml_files))
print("Missing images:", len(missing_images))
print("Missing dimensions:", len(missing_dimensions))
print("Invalid bounding boxes:", len(invalid_boxes))

print("\nClass distribution:")
for name, count in class_counts.items():
    print(name, ":", count)

print("\nUnmatched image examples:", missing_images[:10])
print("Invalid box examples:", invalid_boxes[:10])
