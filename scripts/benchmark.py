
import cv2
import time
from ultralytics import YOLO

VIDEO_PATH = "test_videos/3.mp4"
MODELS = ["yolo11n.pt", "yolo26n.pt"]

CLASSES = [0, 2, 3, 5, 7]
CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

IMGSZ = 640
CONF = 0.20
MAX_FRAMES = None

for model_path in MODELS:

    print(f"\nTesting: {model_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video:", VIDEO_PATH)
        break

    detection_counts = {name: 0 for name in CLASS_NAMES.values()}

    frame_count = 0
    inference_time = 0

    while frame_count < MAX_FRAMES:

        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        result = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF,
            classes=CLASSES,
            device="cpu",
            verbose=False
        )[0]

        inference_time += time.perf_counter() - start

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                name = CLASS_NAMES.get(class_id)

                if name:
                    detection_counts[name] += 1

        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()

    fps = frame_count / inference_time if inference_time else 0

    print("\nRESULTS")
    print("Model:", model_path)
    print("Frames:", frame_count)
    print("Inference time:", round(inference_time, 2), "seconds")
    print("Inference FPS:", round(fps, 2))
    print("Detection totals:", detection_counts)

print("\nBenchmark completed.")
