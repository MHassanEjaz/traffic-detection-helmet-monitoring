
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load fine-tuned helmet model
model = YOLO("helmet_finetuned.pt")

# Input and output paths
input_video = "test_videos/1029.mp4"
output_video = "helmet_test_1029.mp4"

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total frames: {total_frames}")

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_video, fourcc, fps, (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Helmet detection
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    # Draw predictions
    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    frame_number += 1

    if frame_number % 100 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")

cap.release()
writer.release()

print("\nProcessing complete!")
print(f"Processed frames: {frame_number}")
print(f"Output saved: {output_video}")
