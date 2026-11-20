
# import streamlit as st
# import cv2
# import os
# import tempfile
# import subprocess
# import imageio_ffmpeg

# from pathlib import Path
# from collections import defaultdict, deque
# from ultralytics import YOLO


# # ==================================================
# # PROJECT PATHS
# # ==================================================

# ROOT = Path(__file__).resolve().parents[1]

# MODEL_PATH = ROOT / "models" / "yolo26n.pt"
# TRACKER_PATH = ROOT / "trackers" / "bytetrack_custom.yaml"

# HELMET_MODELS = {
#     "Original Helmet Model": ROOT / "models" / "helmet.pt",
#     "Fine-tuned Helmet Model": ROOT / "models" / "helmet_finetuned.pt",
#     "RideSafe Helmet Model": ROOT / "models" / "helmet_ridesafe_best.pt"
# }


# # ==================================================
# # CONFIGURATION
# # ==================================================

# CONFIDENCE_THRESHOLD = 0.15
# IOU_THRESHOLD = 0.50
# IMAGE_SIZE = 640

# HELMET_CONFIDENCE = 0.10
# HELMET_IMAGE_SIZE = 1280

# # Helmet detection every 3rd frame
# HELMET_DETECTION_INTERVAL = 3

# # Retain detections for skipped frames
# HELMET_MAX_AGE = 2

# LINE_POSITION_RATIO = 0.50


# # ==================================================
# # CLASS NAMES AND COLORS
# # ==================================================

# CLASS_NAMES = {
#     0: "Person",
#     2: "Car",
#     3: "Motorcycle",
#     5: "Bus",
#     7: "Truck"
# }

# CLASS_COLORS = {
#     0: (255, 255, 0),
#     2: (0, 255, 0),
#     3: (255, 0, 255),
#     5: (0, 165, 255),
#     7: (255, 100, 0)
# }

# HELMET_CLASS_NAMES = {
#     0: "With Helmet",
#     1: "Without Helmet"
# }

# HELMET_COLORS = {
#     0: (0, 255, 0),
#     1: (0, 0, 255)
# }


# # ==================================================
# # STREAMLIT PAGE
# # ==================================================

# st.set_page_config(
#     page_title="Traffic Detection and Helmet Monitoring",
#     page_icon="🚦",
#     layout="wide"
# )

# st.title("Traffic Detection and Helmet Monitoring")

# st.write(
#     "Upload a traffic video to detect vehicles, "
#     "count traffic movement and identify helmet usage."
# )


# # ==================================================
# # MODEL LOADING
# # ==================================================

# @st.cache_resource
# def load_model(model_path):
#     return YOLO(str(model_path))


# # ==================================================
# # IOU CALCULATION
# # ==================================================

# def calculate_iou(box1, box2):

#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])

#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     intersection = max(0, x2 - x1) * max(0, y2 - y1)

#     area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
#     area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

#     union = area1 + area2 - intersection

#     if union <= 0:
#         return 0

#     return intersection / union


# # ==================================================
# # FULL-FRAME HELMET DETECTION
# # ==================================================

# def detect_frame_helmets(frame, helmet_model):

#     results = helmet_model.predict(
#         frame,
#         conf=HELMET_CONFIDENCE,
#         iou=0.45,
#         imgsz=HELMET_IMAGE_SIZE,
#         verbose=False,
#         device="cpu"
#     )[0]

#     detections = []

#     if results.boxes is None:
#         return detections

#     for box in results.boxes:

#         class_id = int(box.cls[0].item())

#         if class_id not in HELMET_CLASS_NAMES:
#             continue

#         confidence = float(box.conf[0].item())

#         x1, y1, x2, y2 = (
#             box.xyxy[0]
#             .cpu()
#             .numpy()
#             .astype(int)
#         )

#         detections.append({
#             "class_id": class_id,
#             "confidence": confidence,
#             "box": (x1, y1, x2, y2),
#             "age": 0
#         })

#     return detections


# # ==================================================
# # HELMET DETECTION PERSISTENCE
# # ==================================================

# def update_helmet_tracks(current_detections, previous_tracks):

#     updated_tracks = []
#     matched_previous = set()

#     for current in current_detections:

#         best_iou = 0
#         best_index = None

#         for index, previous in enumerate(previous_tracks):

#             if index in matched_previous:
#                 continue

#             if current["class_id"] != previous["class_id"]:
#                 continue

#             overlap = calculate_iou(
#                 current["box"],
#                 previous["box"]
#             )

#             if overlap > best_iou:
#                 best_iou = overlap
#                 best_index = index

#         if best_index is not None and best_iou >= 0.25:

#             previous = previous_tracks[best_index]
#             matched_previous.add(best_index)

#             old_box = previous["box"]
#             new_box = current["box"]

#             smooth_box = tuple(
#                 int(0.70 * new + 0.30 * old)
#                 for new, old in zip(new_box, old_box)
#             )

#             current["box"] = smooth_box

#         updated_tracks.append(current)

#     # Retain temporarily missed detections
#     for index, previous in enumerate(previous_tracks):

#         if index in matched_previous:
#             continue

#         previous_age = previous["age"] + 1

#         if previous_age <= HELMET_MAX_AGE:

#             retained = previous.copy()
#             retained["age"] = previous_age

#             updated_tracks.append(retained)

#     return updated_tracks


# # ==================================================
# # DRAW HELMET DETECTIONS
# # ==================================================

# def draw_helmet_detections(frame, detections):

#     for detection in detections:

#         class_id = detection["class_id"]
#         confidence = detection["confidence"]

#         x1, y1, x2, y2 = detection["box"]

#         label = (
#             f"{HELMET_CLASS_NAMES[class_id]} "
#             f"{confidence:.2f}"
#         )

#         color = HELMET_COLORS[class_id]

#         # Thick bounding box
#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             color,
#             4
#         )

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.75
#         thickness = 2

#         (text_width, text_height), baseline = cv2.getTextSize(
#             label,
#             font,
#             font_scale,
#             thickness
#         )

#         label_y = max(y1, text_height + 15)

#         # Filled label background
#         cv2.rectangle(
#             frame,
#             (x1, label_y - text_height - 12),
#             (x1 + text_width + 12, label_y + baseline),
#             color,
#             -1
#         )

#         cv2.putText(
#             frame,
#             label,
#             (x1 + 6, label_y - 5),
#             font,
#             font_scale,
#             (255, 255, 255),
#             thickness,
#             cv2.LINE_AA
#         )


# # ==================================================
# # RIDER ASSOCIATION
# # ==================================================

# def is_motorcycle_rider(person_box, motorcycle_boxes):

#     px1, py1, px2, py2 = person_box

#     person_center_x = (px1 + px2) / 2
#     person_bottom_y = py2

#     for bx1, by1, bx2, by2 in motorcycle_boxes:

#         bike_center_x = (bx1 + bx2) / 2
#         bike_width = bx2 - bx1

#         horizontal_match = (
#             bx1 - 0.25 * bike_width
#             <= person_center_x
#             <= bx2 + 0.25 * bike_width
#         )

#         vertical_match = (
#             person_bottom_y >= by1
#             and py1 <= by2
#         )

#         center_match = (
#             abs(person_center_x - bike_center_x)
#             <= max(bike_width * 0.75, 30)
#         )

#         if horizontal_match and (vertical_match or center_match):
#             return True

#     return False


# # ==================================================
# # VIDEO PROCESSING
# # ==================================================

# def process_video(input_path, output_path, helmet_model_path):

#     vehicle_model = load_model(MODEL_PATH)
#     helmet_model = load_model(helmet_model_path)

#     cap = cv2.VideoCapture(str(input_path))

#     if not cap.isOpened():
#         raise RuntimeError("Could not open input video.")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     fps = cap.get(cv2.CAP_PROP_FPS)

#     if fps <= 0:
#         fps = 25

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     if width <= 0 or height <= 0:
#         cap.release()
#         raise RuntimeError("Invalid video dimensions.")

#     line_y = int(height * LINE_POSITION_RATIO)

#     temp_output = str(output_path).replace(
#         ".mp4",
#         "_temp.mp4"
#     )

#     writer = cv2.VideoWriter(
#         temp_output,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height)
#     )

#     if not writer.isOpened():
#         cap.release()
#         raise RuntimeError("Could not create output video.")

#     track_history = defaultdict(lambda: deque(maxlen=2))
#     counted_ids = set()

#     helmet_tracks = []

#     counts = {
#         name: {
#             "Up": 0,
#             "Down": 0
#         }
#         for name in CLASS_NAMES.values()
#     }

#     frame_number = 0

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     try:

#         while True:

#             ret, frame = cap.read()

#             if not ret:
#                 break

#             frame_number += 1

#             # Keep original frame for helmet model
#             original_frame = frame.copy()

#             # ------------------------------------------
#             # HELMET DETECTION EVERY 3RD FRAME
#             # ------------------------------------------

#             if (
#                 frame_number - 1
#             ) % HELMET_DETECTION_INTERVAL == 0:

#                 current_helmet_detections = detect_frame_helmets(
#                     original_frame,
#                     helmet_model
#                 )

#             else:
#                 current_helmet_detections = []

#             helmet_tracks = update_helmet_tracks(
#                 current_helmet_detections,
#                 helmet_tracks
#             )

#             # ------------------------------------------
#             # VEHICLE AND PERSON DETECTION
#             # ------------------------------------------

#             results = vehicle_model.track(
#                 frame,
#                 persist=True,
#                 tracker=str(TRACKER_PATH),
#                 conf=CONFIDENCE_THRESHOLD,
#                 iou=IOU_THRESHOLD,
#                 imgsz=IMAGE_SIZE,
#                 classes=list(CLASS_NAMES.keys()),
#                 verbose=False,
#                 device="cpu"
#             )[0]

#             person_boxes = []
#             motorcycle_boxes = []

#             if results.boxes is not None:

#                 for box in results.boxes:

#                     class_id = int(box.cls[0].item())

#                     if class_id not in CLASS_NAMES:
#                         continue

#                     x1, y1, x2, y2 = (
#                         box.xyxy[0]
#                         .cpu()
#                         .numpy()
#                         .astype(int)
#                     )

#                     confidence = float(box.conf[0].item())

#                     class_name = CLASS_NAMES[class_id]
#                     color = CLASS_COLORS[class_id]

#                     if class_id == 0:
#                         person_boxes.append(
#                             (x1, y1, x2, y2)
#                         )

#                     elif class_id == 3:
#                         motorcycle_boxes.append(
#                             (x1, y1, x2, y2)
#                         )

#                     track_id = None

#                     if box.id is not None:
#                         track_id = int(box.id[0].item())

#                     label = f"{class_name} {confidence:.2f}"

#                     if track_id is not None:
#                         label += f" ID:{track_id}"

#                     cv2.rectangle(
#                         frame,
#                         (x1, y1),
#                         (x2, y2),
#                         color,
#                         3
#                     )

#                     cv2.putText(
#                         frame,
#                         label,
#                         (x1, max(y1 - 8, 20)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.65,
#                         color,
#                         2,
#                         cv2.LINE_AA
#                     )

#                     # ----------------------------------
#                     # LINE CROSSING COUNTING
#                     # ----------------------------------

#                     if track_id is not None:

#                         center_x = int((x1 + x2) / 2)
#                         center_y = int(y2)

#                         track_history[track_id].append(
#                             (center_x, center_y)
#                         )

#                         if len(track_history[track_id]) == 2:

#                             previous_y = track_history[track_id][0][1]
#                             current_y = track_history[track_id][1][1]

#                             if track_id not in counted_ids:

#                                 if previous_y < line_y <= current_y:

#                                     counts[class_name]["Down"] += 1
#                                     counted_ids.add(track_id)

#                                 elif previous_y > line_y >= current_y:

#                                     counts[class_name]["Up"] += 1
#                                     counted_ids.add(track_id)

#             # ------------------------------------------
#             # RIDER INFORMATION
#             # ------------------------------------------

#             rider_count = sum(
#                 1
#                 for person_box in person_boxes
#                 if is_motorcycle_rider(
#                     person_box,
#                     motorcycle_boxes
#                 )
#             )

#             # ------------------------------------------
#             # DRAW HELMET DETECTIONS
#             # ------------------------------------------

#             draw_helmet_detections(
#                 frame,
#                 helmet_tracks
#             )

#             # ------------------------------------------
#             # COUNTING LINE
#             # ------------------------------------------

#             cv2.line(
#                 frame,
#                 (0, line_y),
#                 (width, line_y),
#                 (0, 0, 255),
#                 3
#             )

#             cv2.putText(
#                 frame,
#                 "COUNTING LINE",
#                 (10, max(line_y - 10, 25)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75,
#                 (0, 0, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             # ------------------------------------------
#             # COUNTING INFORMATION
#             # ------------------------------------------

#             y_offset = 30

#             for class_name, directions in counts.items():

#                 text = (
#                     f"{class_name}: "
#                     f"Up={directions['Up']} "
#                     f"Down={directions['Down']}"
#                 )

#                 cv2.putText(
#                     frame,
#                     text,
#                     (10, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.60,
#                     (255, 255, 255),
#                     2,
#                     cv2.LINE_AA
#                 )

#                 y_offset += 28

#             cv2.putText(
#                 frame,
#                 f"Motorcycle Riders: {rider_count}",
#                 (10, y_offset + 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             writer.write(frame)

#             # ------------------------------------------
#             # PROGRESS
#             # ------------------------------------------

#             if frame_number % 5 == 0 or frame_number == total_frames:

#                 if total_frames > 0:

#                     progress_bar.progress(
#                         min(frame_number / total_frames, 1.0)
#                     )

#                 status_text.text(
#                     f"Processing frame: {frame_number}/{total_frames}"
#                 )

#     finally:

#         cap.release()
#         writer.release()

#     # ==================================================
#     # H.264 VIDEO CONVERSION
#     # ==================================================

#     ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

#     try:

#         subprocess.run(
#             [
#                 ffmpeg_exe,
#                 "-y",
#                 "-i", temp_output,
#                 "-vcodec", "libx264",
#                 "-pix_fmt", "yuv420p",
#                 "-preset", "ultrafast",
#                 "-crf", "23",
#                 "-acodec", "aac",
#                 str(output_path)
#             ],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )

#         os.remove(temp_output)

#     except Exception:
#         os.replace(temp_output, output_path)

#     progress_bar.progress(1.0)

#     status_text.success(
#         f"Completed: {frame_number}/{total_frames} frames"
#     )

#     return counts, frame_number


# # ==================================================
# # USER INTERFACE
# # ==================================================

# st.subheader("Helmet Model Selection")

# selected_model_name = st.selectbox(
#     "Select helmet detection model",
#     list(HELMET_MODELS.keys()),
#     index=2
# )

# selected_model_path = HELMET_MODELS[selected_model_name]

# st.caption(
#     f"Selected model: {selected_model_path.name}"
# )

# uploaded_file = st.file_uploader(
#     "Upload traffic video",
#     type=["mp4", "avi", "mov", "mkv"]
# )


# # ==================================================
# # PROCESS VIDEO
# # ==================================================

# if uploaded_file is not None:

#     st.subheader("Input Video")
#     st.video(uploaded_file)

#     if st.button(
#         "Process Video",
#         type="primary",
#         use_container_width=True
#     ):

#         missing_files = []

#         for path in [
#             MODEL_PATH,
#             TRACKER_PATH,
#             selected_model_path
#         ]:

#             if not path.exists():
#                 missing_files.append(str(path))

#         if missing_files:

#             st.error("Required files are missing:")

#             for path in missing_files:
#                 st.write(path)

#         else:

#             with tempfile.TemporaryDirectory() as temp_dir:

#                 input_path = Path(temp_dir) / uploaded_file.name

#                 output_path = (
#                     Path(temp_dir) /
#                     "processed_traffic_video.mp4"
#                 )

#                 with open(input_path, "wb") as file:
#                     file.write(uploaded_file.getbuffer())

#                 try:

#                     st.subheader("Processing")

#                     counts, processed_frames = process_video(
#                         input_path,
#                         output_path,
#                         selected_model_path
#                     )

#                     if output_path.exists():

#                         video_bytes = output_path.read_bytes()

#                         st.subheader("Processed Video")

#                         st.video(video_bytes)

#                         st.download_button(
#                             label="Download Processed Video",
#                             data=video_bytes,
#                             file_name="processed_traffic_video.mp4",
#                             mime="video/mp4",
#                             use_container_width=True
#                         )

#                         st.subheader("Traffic Counting Results")

#                         for class_name, directions in counts.items():

#                             st.write(
#                                 f"**{class_name}:** "
#                                 f"Up = {directions['Up']} | "
#                                 f"Down = {directions['Down']}"
#                             )

#                         st.success(
#                             f"Successfully processed {processed_frames} frames."
#                         )

#                     else:

#                         st.error(
#                             "Processed video was not generated."
#                         )

#                 except Exception as error:

#                     st.error(
#                         f"Processing failed: {error}"
#                     )

# else:

#     st.info("Upload a video to get started.")




import sys
from pathlib import Path

# Project root ko Python path mein add karo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import requests
import streamlit as st

from app.core.config import HELMET_MODELS




# ==================================================
# CONFIGURATION
# ==================================================

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Traffic Detection and Helmet Monitoring",
    page_icon="🚦",
    layout="wide"
)


# ==================================================
# PAGE HEADER
# ==================================================

st.title("Traffic Detection and Helmet Monitoring")

st.write(
    "Upload a traffic video to detect vehicles, "
    "count traffic movement and identify helmet usage."
)


# ==================================================
# HELMET MODEL SELECTION
# ==================================================

st.subheader("Helmet Model Selection")

selected_model_name = st.selectbox(
    "Select helmet detection model",
    list(HELMET_MODELS.keys()),
    index=2
)

st.caption(
    f"Selected model: {HELMET_MODELS[selected_model_name].name}"
)


# ==================================================
# VIDEO UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload traffic video",
    type=["mp4", "avi", "mov", "mkv", "webm"]
)


# ==================================================
# API HELPERS
# ==================================================

def upload_video(video_file, model_name):

    files = {
        "file": (
            video_file.name,
            video_file.getvalue(),
            "application/octet-stream"
        )
    }

    data = {
        "helmet_model": model_name
    }

    response = requests.post(
        f"{API_BASE_URL}/api/process",
        files=files,
        data=data,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def get_job_status(job_id):

    response = requests.get(
        f"{API_BASE_URL}/api/jobs/{job_id}/status",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_job_result(job_id):

    response = requests.get(
        f"{API_BASE_URL}/api/jobs/{job_id}/result",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_processed_video(job_id):

    response = requests.get(
        f"{API_BASE_URL}/api/jobs/{job_id}/video",
        timeout=120
    )

    response.raise_for_status()

    return response.content


# ==================================================
# PROCESS VIDEO
# ==================================================

if uploaded_file is not None:

    st.subheader("Input Video")

    st.video(uploaded_file)

    if st.button(
        "Process Video",
        type="primary",
        use_container_width=True
    ):

        try:

            # --------------------------------------
            # UPLOAD VIDEO TO FASTAPI
            # --------------------------------------

            with st.spinner("Uploading video to backend..."):

                upload_response = upload_video(
                    uploaded_file,
                    selected_model_name
                )

            job_id = upload_response["job_id"]

            st.success("Video uploaded successfully!")

            st.write(f"**Job ID:** `{job_id}`")

            # --------------------------------------
            # PROCESSING PROGRESS
            # --------------------------------------

            st.subheader("Processing")

            progress_bar = st.progress(0)

            status_text = st.empty()

            while True:

                status = get_job_status(job_id)

                progress = status["progress"]

                progress_bar.progress(
                    min(int(progress), 100)
                )

                status_text.info(
                    f'{status["message"]} '
                    f'({progress:.1f}%)'
                )

                if status["status"] == "completed":
                    break

                if status["status"] == "failed":

                    raise RuntimeError(
                        status.get("error")
                        or "Backend processing failed."
                    )

                time.sleep(2)

            # --------------------------------------
            # GET PROCESSING RESULTS
            # --------------------------------------

            result = get_job_result(job_id)

            st.success(
                f'Successfully processed '
                f'{result["processed_frames"]} frames.'
            )

            # --------------------------------------
            # GET PROCESSED VIDEO
            # --------------------------------------

            with st.spinner("Loading processed video..."):

                video_bytes = get_processed_video(job_id)

            st.subheader("Processed Video")

            st.video(video_bytes)

            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_traffic_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

            # --------------------------------------
            # TRAFFIC COUNTING RESULTS
            # --------------------------------------

            st.subheader("Traffic Counting Results")

            counts = result["counts"]

            for class_name, directions in counts.items():

                st.write(
                    f"**{class_name}:** "
                    f"Up = {directions['Up']} | "
                    f"Down = {directions['Down']}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Please make sure the backend is running."
            )

        except requests.exceptions.RequestException as error:

            st.error(f"API request failed: {error}")

        except Exception as error:

            st.error(f"Processing failed: {error}")

else:

    st.info("Upload a video to get started.")
