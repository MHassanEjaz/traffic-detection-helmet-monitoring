
import cv2
import time
from collections import defaultdict
from ultralytics import YOLO

VIDEO_PATH = "test_videos/3.mp4"

MODELS = [
    "yolo11n.pt",
    "yolo26n.pt"
]

TRACKERS = [
    "trackers/bytetrack_custom.yaml",
    "botsort.yaml"
]

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

    for tracker_path in TRACKERS:

        print("\n" + "=" * 50)
        print("Model:", model_path)
        print("Tracker:", tracker_path)

        model = YOLO(model_path)
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print("Could not open video.")
            continue

        unique_ids = defaultdict(set)
        frame_count = 0
        processing_time = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            start = time.perf_counter()

            result = model.track(
                frame,
                persist=True,
                tracker=tracker_path,
                classes=CLASSES,
                conf=CONF,
                iou=0.50,
                imgsz=IMGSZ,
                device="cpu",
                verbose=False
            )[0]

            processing_time += time.perf_counter() - start

            if result.boxes is not None and result.boxes.id is not None:

                ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for track_id, class_id in zip(ids, class_ids):

                    if class_id in CLASS_NAMES:
                        unique_ids[CLASS_NAMES[class_id]].add(track_id)

            frame_count += 1

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")

            if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
                break

        cap.release()

        fps = (
            frame_count / processing_time
            if processing_time > 0 else 0
        )

        print("\nRESULTS")
        print("Frames:", frame_count)
        print("Processing time:", round(processing_time, 2), "seconds")
        print("Tracking FPS:", round(fps, 2))

        total_ids = 0

        for name in CLASS_NAMES.values():

            count = len(unique_ids[name])
            total_ids += count

            print(f"{name}: {count} unique IDs")

        print("Total unique IDs:", total_ids)

print("\nTracking benchmark completed.")
