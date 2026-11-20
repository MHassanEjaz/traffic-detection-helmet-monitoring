
import cv2
from ultralytics import YOLO
from pathlib import Path
import csv

VIDEO_PATH = "test_videos/1029.mp4"

MODELS = {
    "Original": "helmet.pt",
    "Fine-tuned": "helmet_finetuned.pt"
}

FRAME_INTERVAL = 10
PERSON_CONF = 0.25
HELMET_CONF = 0.01
IMAGE_SIZE = 640
CROP_SCALE = 3

output_dir = Path("helmet_model_video_comparison")
output_dir.mkdir(exist_ok=True)

# Load models
person_model = YOLO("yolo26n.pt")

helmet_models = {
    name: YOLO(path)
    for name, path in MODELS.items()
}

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Total frames:", total_frames)

csv_path = output_dir / "comparison_results.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Frame",
        "Person_ID",
        "Model",
        "Class",
        "Confidence"
    ])

    frame_number = 0
    crop_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % FRAME_INTERVAL == 0:

            person_result = person_model.predict(
                frame,
                classes=[0],
                conf=PERSON_CONF,
                imgsz=640,
                verbose=False
            )[0]

            for person_id, box in enumerate(person_result.boxes):

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].cpu().numpy()
                )

                person_h = y2 - y1
                person_w = x2 - x1

                # Upper 50% of person
                crop_y2 = min(
                    frame.shape[0],
                    y1 + int(person_h * 0.50)
                )

                crop = frame[
                    max(0, y1):crop_y2,
                    max(0, x1):min(frame.shape[1], x2)
                ]

                if crop.size == 0:
                    continue

                crop_number += 1

                crop = cv2.resize(
                    crop,
                    None,
                    fx=CROP_SCALE,
                    fy=CROP_SCALE
                )

                crop_file = output_dir / (
                    f"frame_{frame_number}_person_{person_id}.jpg"
                )

                cv2.imwrite(str(crop_file), crop)

                for model_name, model in helmet_models.items():

                    result = model.predict(
                        crop,
                        imgsz=IMAGE_SIZE,
                        conf=HELMET_CONF,
                        verbose=False
                    )[0]

                    for detection in result.boxes:

                        class_id = int(detection.cls[0].item())
                        confidence = float(
                            detection.conf[0].item()
                        )

                        class_name = model.names[class_id]

                        writer.writerow([
                            frame_number,
                            person_id,
                            model_name,
                            class_name,
                            round(confidence, 4)
                        ])

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

cap.release()

print("\nComparison completed.")
print("Total person crops:", crop_number)
print("CSV saved:", csv_path)
