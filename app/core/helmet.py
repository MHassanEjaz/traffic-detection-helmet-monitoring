import cv2

from app.core.config import (
    HELMET_CONFIDENCE,
    HELMET_IMAGE_SIZE,
    HELMET_MAX_AGE
)





HELMET_CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}

HELMET_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def detect_frame_helmets(frame, helmet_model):
    results = helmet_model.predict(
        frame,
        conf=HELMET_CONFIDENCE,
        iou=0.45,
        imgsz=HELMET_IMAGE_SIZE,
        verbose=False,
        device="cpu"
    )[0]

    detections = []

    if results.boxes is None:
        return detections

    for box in results.boxes:
        class_id = int(box.cls[0].item())

        if class_id not in HELMET_CLASS_NAMES:
            continue

        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
            "age": 0
        })

    return detections


def update_helmet_tracks(current_detections, previous_tracks):
    updated_tracks = []
    matched_previous = set()

    for current in current_detections:
        best_iou = 0
        best_index = None

        for index, previous in enumerate(previous_tracks):
            if index in matched_previous:
                continue

            if current["class_id"] != previous["class_id"]:
                continue

            overlap = calculate_iou(
                current["box"],
                previous["box"]
            )

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= 0.25:
            previous = previous_tracks[best_index]
            matched_previous.add(best_index)

            old_box = previous["box"]
            new_box = current["box"]

            smooth_box = tuple(
                int(0.70 * new + 0.30 * old)
                for new, old in zip(new_box, old_box)
            )

            current["box"] = smooth_box

        updated_tracks.append(current)

    # Retain temporarily missed detections
    for index, previous in enumerate(previous_tracks):
        if index in matched_previous:
            continue

        previous_age = previous["age"] + 1

        if previous_age <= HELMET_MAX_AGE:
            retained = previous.copy()
            retained["age"] = previous_age
            updated_tracks.append(retained)

    return updated_tracks


def draw_helmet_detections(frame, detections):
    for detection in detections:
        class_id = detection["class_id"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = detection["box"]

        label = (
            f"{HELMET_CLASS_NAMES[class_id]} "
            f"{confidence:.2f}"
        )

        color = HELMET_COLORS[class_id]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            4
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        label_y = max(y1, text_height + 15)

        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 12),
            (x1 + text_width + 12, label_y + baseline),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 6, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
