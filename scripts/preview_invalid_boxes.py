
import cv2
import os
import xml.etree.ElementTree as ET

IMAGE_DIR = "helmet_dataset/images"
ANNOTATION_DIR = "helmet_dataset/annotations"

FILES = [
    "BikesHelmets103.xml",
    "BikesHelmets140.xml",
    "BikesHelmets279.xml",
    "BikesHelmets75.xml"
]

for filename in FILES:

    xml_path = os.path.join(ANNOTATION_DIR, filename)
    root = ET.parse(xml_path).getroot()

    image_name = root.findtext("filename")
    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_name)
        continue

    height, width = image.shape[:2]

    for obj in root.findall("object"):

        name = obj.findtext("name")
        box = obj.find("bndbox")

        xmin = int(box.findtext("xmin"))
        ymin = int(box.findtext("ymin"))
        xmax = int(box.findtext("xmax"))
        ymax = int(box.findtext("ymax"))

        # Preview only: test the suspected scaling.
        xmin //= 1000
        ymin //= 1000
        xmax //= 1000
        ymax //= 1000

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            name,
            (xmin, max(20, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
