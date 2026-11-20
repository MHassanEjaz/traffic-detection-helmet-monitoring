
import cv2
import os
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "test_videos/1029.mp4"

PERSON_MODEL_PATH = "yolo26n.pt"
ORIGINAL_MODEL_PATH = "helmet.pt"
FINETUNED_MODEL_PATH = "helmet_finetuned.pt"

OUTPUT_DIR = "helmet_crop_50_comparison"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "original_predictions")
FINETUNED_DIR = os.path.join(OUTPUT_DIR, "finetuned_predictions")

FRAME_NUMBERS = [100, 200, 300, 500, 700, 900, 1100, 1300]

PERSON_CONF = 0.25
HELMET_CONF = 0.01

CROP_HEIGHT_RATIO = 0.50
RESIZE_FACTOR = 3

# Dataset class mapping:
# 0 = With Helmet
# 1 = Without Helmet

CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}

# ------------------------------------------------


def create_directories():
    for folder in [CROPS_DIR, ORIGINAL_DIR, FINETUNED_DIR]:
        os.makedirs(folder, exist_ok=True)


def get_upper_person_crop(frame, box):
    x1, y1, x2, y2 = map(int, box)

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    person_height = y2 - y1
    person_width = x2 - x1

    if person_height <= 0 or person_width <= 0:
        return None

    # Take upper 50% of the detected person
    crop_y2 = y1 + int(person_height * CROP_HEIGHT_RATIO)

    crop = frame[y1:crop_y2, x1:x2]

    if crop.size == 0:
        return None

    # Enlarge crop
    crop = cv2.resize(
        crop,
        None,
        fx=RESIZE_FACTOR,
        fy=RESIZE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    return crop


def run_helmet_model(model, crop):
    results = model.predict(
        source=crop,
        conf=HELMET_CONF,
        imgsz=640,
        verbose=False
    )

    return results[0]


def save_prediction_image(result, output_path):
    annotated = result.plot()
    cv2.imwrite(output_path, annotated)


def print_predictions(result, model_name):
    print(f"{model_name}:")

    if len(result.boxes) == 0:
        print("  No detections")
        return 0

    count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = CLASS_NAMES.get(
            class_id,
            f"Unknown class {class_id}"
        )

        print(
            f"  {class_name} - "
            f"Confidence: {confidence:.3f}"
        )

        count += 1

    return count


def main():
    print("Loading models...")

    person_model = YOLO(PERSON_MODEL_PATH)
    original_model = YOLO(ORIGINAL_MODEL_PATH)
    finetuned_model = YOLO(FINETUNED_MODEL_PATH)

    print("All models loaded successfully!")

    create_directories()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video:", VIDEO_PATH)
        return

    total_original_detections = 0
    total_finetuned_detections = 0
    total_person_crops = 0

    for frame_number in FRAME_NUMBERS:

        print(f"\n{'=' * 50}")
        print(f"Processing frame: {frame_number}")
        print(f"{'=' * 50}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        # Detect people
        person_results = person_model.predict(
            source=frame,
            classes=[0],
            conf=PERSON_CONF,
            imgsz=640,
            verbose=False
        )

        person_boxes = person_results[0].boxes

        frame_crop_count = 0

        for person_index, person_box in enumerate(person_boxes):

            crop = get_upper_person_crop(
                frame,
                person_box.xyxy[0].tolist()
            )

            if crop is None:
                continue

            frame_crop_count += 1
            total_person_crops += 1

            crop_name = (
                f"frame_{frame_number}_crop_{frame_crop_count}.jpg"
            )

            crop_path = os.path.join(CROPS_DIR, crop_name)

            cv2.imwrite(crop_path, crop)

            print(f"\n{crop_name}")

            # Original model
            original_result = run_helmet_model(
                original_model,
                crop
            )

            original_count = print_predictions(
                original_result,
                "Original model"
            )

            total_original_detections += original_count

            original_output = os.path.join(
                ORIGINAL_DIR,
                crop_name
            )

            save_prediction_image(
                original_result,
                original_output
            )

            # Fine-tuned model
            finetuned_result = run_helmet_model(
                finetuned_model,
                crop
            )

            finetuned_count = print_predictions(
                finetuned_result,
                "Fine-tuned model"
            )

            total_finetuned_detections += finetuned_count

            finetuned_output = os.path.join(
                FINETUNED_DIR,
                crop_name
            )

            save_prediction_image(
                finetuned_result,
                finetuned_output
            )

        print(
            f"\nPerson crops in frame: {frame_crop_count}"
        )

    cap.release()

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print("Crop height ratio:", CROP_HEIGHT_RATIO)
    print("Resize factor:", RESIZE_FACTOR)
    print("Total person crops:", total_person_crops)
    print(
        "Original model detections:",
        total_original_detections
    )
    print(
        "Fine-tuned model detections:",
        total_finetuned_detections
    )

    print("\nResults saved in:")
    print(OUTPUT_DIR)

    print("\nTesting completed!")


if __name__ == "__main__":
    main()
