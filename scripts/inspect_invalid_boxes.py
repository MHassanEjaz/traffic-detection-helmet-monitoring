
import os
import xml.etree.ElementTree as ET

ANNOTATION_DIR = "helmet_dataset/annotations"

for filename in os.listdir(ANNOTATION_DIR):

    if not filename.endswith(".xml"):
        continue

    path = os.path.join(ANNOTATION_DIR, filename)
    root = ET.parse(path).getroot()

    size = root.find("size")

    if size is None:
        continue

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    for obj in root.findall("object"):

        name = obj.findtext("name", "Unknown")
        box = obj.find("bndbox")

        if box is None:
            print(filename, "| Missing bounding box")
            continue

        xmin = int(box.findtext("xmin", 0))
        ymin = int(box.findtext("ymin", 0))
        xmax = int(box.findtext("xmax", 0))
        ymax = int(box.findtext("ymax", 0))

        valid = (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        )

        if not valid:
            print(
                f"{filename} | {name} | "
                f"Image: {width}x{height} | "
                f"Box: ({xmin}, {ymin}, {xmax}, {ymax})"
            )
