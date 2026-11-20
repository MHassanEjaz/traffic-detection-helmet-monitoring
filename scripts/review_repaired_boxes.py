
import os
import cv2
import xml.etree.ElementTree as ET

ANNOTATION_DIR = "helmet_dataset/annotations"
IMAGE_DIR = "helmet_dataset/images"

FILES = [
    "BikesHelmets103.xml",
    "BikesHelmets140.xml",
    "BikesHelmets205.xml",
    "BikesHelmets279.xml",
    "BikesHelmets326.xml",
    "BikesHelmets343.xml",
    "BikesHelmets441.xml",
    "BikesHelmets444.xml",
    "BikesHelmets530.xml",
    "BikesHelmets616.xml",
    "BikesHelmets671.xml",
    "BikesHelmets706.xml",
    "BikesHelmets75.xml",
    "BikesHelmets764.xml",
    "BikesHelmets80.xml"
]

for filename in FILES:
    root = ET.parse(os.path.join(ANNOTATION_DIR, filename)).getroot()

    image_name = root.findtext("filename")
    image = cv2.imread(os.path.join(IMAGE_DIR, image_name))

    if image is None:
        print("Image missing:", image_name)
        continue

    height, width = image.shape[:2]

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        coords = [
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax"))
        ]

        x1, y1, x2, y2 = coords

        if x2 > width or y2 > height or x1 < 0 or y1 < 0:
            x1, y1, x2, y2 = [v / 1000 for v in coords]

            x1, y1, x2, y2 = map(
                lambda v: int(round(v)),
                (x1, y1, x2, y2)
            )

            label = obj.findtext("name")

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                image, label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

    cv2.imshow(filename, image)
    print("Reviewing:", filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
