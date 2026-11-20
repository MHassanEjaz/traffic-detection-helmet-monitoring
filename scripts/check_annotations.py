
import os
import xml.etree.ElementTree as ET
from collections import Counter

ANNOTATIONS_DIR = "helmet_dataset/annotations"

class_counts = Counter()
xml_count = 0
missing_files = []

for filename in os.listdir(ANNOTATIONS_DIR):

    if not filename.endswith(".xml"):
        continue

    xml_count += 1
    path = os.path.join(ANNOTATIONS_DIR, filename)

    try:
        tree = ET.parse(path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.findtext("name", default="Unknown").strip()
            class_counts[name] += 1

    except ET.ParseError:
        missing_files.append(filename)

print("Total XML files:", xml_count)
print("\nClass distribution:")

for name, count in class_counts.items():
    print(f"{name}: {count}")

print("\nInvalid XML files:", len(missing_files))
print(missing_files[:10])
