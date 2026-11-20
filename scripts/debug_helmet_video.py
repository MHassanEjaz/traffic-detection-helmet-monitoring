
import cv2
from ultralytics import YOLO

model = YOLO("helmet_finetuned.pt")
cap = cv2.VideoCapture("test_videos/1029.mp4")

if not cap.isOpened():
    raise RuntimeError("Could not open video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Test 10 frames at different positions
frame_positions = [
    0, 100, 200, 300, 400,
    500, 700, 900, 1100, 1300
]

for position in frame_positions:
    if position >= total_frames:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {position}")
        continue

    results = model.predict(
        source=frame,
        imgsz=960,
        conf=0.01,
        verbose=False
    )

    result = results[0]

    print(f"\nFrame {position}:")
    print("Detections:", len(result.boxes))

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Class: {model.names[class_id]}, "
            f"Confidence: {confidence:.3f}"
        )

cap.release()
