
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

ANNOTATION_DIR = "helmet_dataset/annotations"

class_counts = defaultdict(int)
repaired = []
skipped = []
valid = 0

for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith(".xml"):
        continue

    path = os.path.join(ANNOTATION_DIR, filename)
    root = ET.parse(path).getroot()

    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")

        coords = [
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax"))
        ]

        xmin, ymin, xmax, ymax = coords

        if (
            xmin >= 0 and ymin >= 0
            and xmax <= width and ymax <= height
            and xmin < xmax and ymin < ymax
        ):
            valid += 1
            class_counts[name] += 1
            continue

        # Try 1000x scaling only when original coordinates are invalid
        scaled = [v / 1000 for v in coords]
        sx1, sy1, sx2, sy2 = scaled

        if (
            sx1 >= 0 and sy1 >= 0
            and sx2 <= width and sy2 <= height
            and sx1 < sx2 and sy1 < sy2
        ):
            repaired.append({
                "file": filename,
                "class": name,
                "box": tuple(round(v, 2) for v in scaled)
            })
        else:
            skipped.append({
                "file": filename,
                "class": name,
                "original": coords
            })

print("Already valid annotations:", valid)
print("Potentially repairable annotations:", len(repaired))
print("Skipped annotations:", len(skipped))

print("\nPOTENTIAL REPAIRS:")
for item in repaired:
    print(item)

print("\nSKIPPED:")
for item in skipped:
    print(item)

print("\nCLASS COUNTS (original valid boxes):")
for name, count in class_counts.items():
    print(name, count)
