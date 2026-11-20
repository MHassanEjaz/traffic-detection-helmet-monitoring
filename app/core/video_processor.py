
import os
import subprocess

import cv2
import imageio_ffmpeg

from pathlib import Path

from app.core.config import (
    MODEL_PATH,
    TRACKER_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    HELMET_DETECTION_INTERVAL,
    LINE_POSITION_RATIO,
)

from app.core.model import load_model

from app.core.detection import (
    CLASS_NAMES,
    CLASS_COLORS,
    is_motorcycle_rider,
)

from app.core.helmet import (
    detect_frame_helmets,
    update_helmet_tracks,
    draw_helmet_detections,
)

from app.core.tracking import TrafficCounter


def process_video(
    input_path,
    output_path,
    helmet_model_path,
    progress_callback=None
):
    """
    Process traffic video using vehicle detection,
    ByteTrack tracking, helmet detection and line counting.

    Args:
        input_path: Input video path.
        output_path: Processed video output path.
        helmet_model_path: Selected helmet model path.
        progress_callback: Optional callback(progress, message).

    Returns:
        counts, processed_frames
    """

    vehicle_model = load_model(str(MODEL_PATH))
    helmet_model = load_model(str(helmet_model_path))

    input_path = Path(input_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")

    line_y = int(height * LINE_POSITION_RATIO)

    temp_output = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    counter = TrafficCounter(
        CLASS_NAMES,
        line_position_ratio=LINE_POSITION_RATIO
    )

    helmet_tracks = []

    frame_number = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            original_frame = frame.copy()

            # ------------------------------------------
            # HELMET DETECTION EVERY 3RD FRAME
            # ------------------------------------------

            if (
                frame_number - 1
            ) % HELMET_DETECTION_INTERVAL == 0:

                current_helmet_detections = detect_frame_helmets(
                    original_frame,
                    helmet_model
                )

            else:
                current_helmet_detections = []

            helmet_tracks = update_helmet_tracks(
                current_helmet_detections,
                helmet_tracks
            )

            # ------------------------------------------
            # VEHICLE AND PERSON DETECTION
            # ------------------------------------------

            results = vehicle_model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_PATH),
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                classes=list(CLASS_NAMES.keys()),
                verbose=False,
                device="cpu"
            )[0]

            person_boxes = []
            motorcycle_boxes = []

            if results.boxes is not None:

                for box in results.boxes:

                    class_id = int(box.cls[0].item())

                    if class_id not in CLASS_NAMES:
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    confidence = float(box.conf[0].item())

                    class_name = CLASS_NAMES[class_id]
                    color = CLASS_COLORS[class_id]

                    if class_id == 0:
                        person_boxes.append(
                            (x1, y1, x2, y2)
                        )

                    elif class_id == 3:
                        motorcycle_boxes.append(
                            (x1, y1, x2, y2)
                        )

                    track_id = None

                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    label = f"{class_name} {confidence:.2f}"

                    if track_id is not None:
                        label += f" ID:{track_id}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    # ----------------------------------
                    # LINE CROSSING COUNTING
                    # ----------------------------------

                    if track_id is not None:

                        center_x = int((x1 + x2) / 2)
                        center_y = int(y2)

                        counter.update(
                            track_id,
                            class_name,
                            center_x,
                            center_y,
                            line_y
                        )

            # ------------------------------------------
            # RIDER INFORMATION
            # ------------------------------------------

            rider_count = sum(
                1
                for person_box in person_boxes
                if is_motorcycle_rider(
                    person_box,
                    motorcycle_boxes
                )
            )

            # ------------------------------------------
            # DRAW HELMET DETECTIONS
            # ------------------------------------------

            draw_helmet_detections(
                frame,
                helmet_tracks
            )

            # ------------------------------------------
            # COUNTING LINE
            # ------------------------------------------

            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "COUNTING LINE",
                (10, max(line_y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------
            # COUNTING INFORMATION
            # ------------------------------------------

            y_offset = 30

            for class_name, directions in counter.counts.items():

                text = (
                    f"{class_name}: "
                    f"Up={directions['Up']} "
                    f"Down={directions['Down']}"
                )

                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                y_offset += 28

            cv2.putText(
                frame,
                f"Motorcycle Riders: {rider_count}",
                (10, y_offset + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

            # ------------------------------------------
            # PROGRESS CALLBACK
            # ------------------------------------------

            if (
                frame_number % 5 == 0
                or frame_number == total_frames
            ):

                progress = (
                    min(frame_number / total_frames, 1.0)
                    if total_frames > 0
                    else 0.0
                )

                message = (
                    f"Processing frame: "
                    f"{frame_number}/{total_frames}"
                )

                if progress_callback is not None:
                    progress_callback(progress, message)

    finally:

        cap.release()
        writer.release()

    # ----------------------------------------------
    # H.264 VIDEO CONVERSION
    # ----------------------------------------------

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", str(temp_output),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "23",
                "-acodec", "aac",
                str(output_path)
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        os.remove(temp_output)

    except Exception:

        os.replace(temp_output, output_path)

    if progress_callback is not None:

        progress_callback(
            1.0,
            f"Completed: {frame_number}/{total_frames} frames"
        )

    return counter.counts, frame_number
