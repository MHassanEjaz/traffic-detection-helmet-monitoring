
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO("yolo11n.pt")
class_list = model.names

# Only detect traffic vehicles
vehicle_classes = {
    "car", "bus", "truck", "motorcycle"
}

# Open the video file
cap = cv2.VideoCapture("test_videos/3.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    raise SystemExit

# Define line positions for counting
line_y_red = 298
line_y_blue = line_y_red + 100

# Variables to store counting information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Dictionaries to count vehicles by class
count_red_to_blue = defaultdict(int)
count_blue_to_red = defaultdict(int)

# Store the first line crossed by each tracking ID
crossed_red_first = set()
crossed_blue_first = set()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        classes=[2, 3, 5, 7]
    )

    result = results[0]
    boxes = result.boxes

    # Draw the counting lines
    cv2.line(
        frame, (190, line_y_red), (850, line_y_red),
        (0, 0, 255), 3
    )
    cv2.putText(
        frame, "Red Line", (190, line_y_red - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )

    cv2.line(
        frame, (27, line_y_blue), (960, line_y_blue),
        (255, 0, 0), 3
    )
    cv2.putText(
        frame, "Blue Line", (27, line_y_blue - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )

    # Process detected objects if tracking IDs are available
    if boxes is not None and boxes.id is not None:
        coordinates = boxes.xyxy.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        class_indices = boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(
            coordinates, track_ids, class_indices
        ):
            x1, y1, x2, y2 = map(int, box)

            # Calculate the center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get the detected class name
            class_name = class_list[class_idx]

            # Safety filter for traffic vehicle classes
            if class_name not in vehicle_classes:
                continue

            # Draw bounding box and tracking ID
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID: {track_id} {class_name}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )

            # Check if the object crosses the red line
            if abs(cy - line_y_red) <= 5:
                crossed_red_first.add(track_id)

            # Check if the object crosses the blue line
            if abs(cy - line_y_blue) <= 5:
                crossed_blue_first.add(track_id)

            # Count red-to-blue movement (downward)
            if (
                track_id in crossed_red_first
                and abs(cy - line_y_blue) <= 5
                and track_id not in counted_ids_red_to_blue
            ):
                counted_ids_red_to_blue.add(track_id)
                count_red_to_blue[class_name] += 1

            # Count blue-to-red movement (upward)
            if (
                track_id in crossed_blue_first
                and abs(cy - line_y_red) <= 5
                and track_id not in counted_ids_blue_to_red
            ):
                counted_ids_blue_to_red.add(track_id)
                count_blue_to_red[class_name] += 1

    # Display downward counts
    y_offset = 30

    for class_name in ["car", "bus", "truck", "motorcycle"]:
        count = count_red_to_blue[class_name]

        cv2.putText(
            frame,
            f"{class_name} (Down): {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2, cv2.LINE_AA
        )
        y_offset += 30

    # Display upward counts
    y_offset += 10

    for class_name in ["car", "bus", "truck", "motorcycle"]:
        count = count_blue_to_red[class_name]

        cv2.putText(
            frame,
            f"{class_name} (Up): {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2, cv2.LINE_AA
        )
        y_offset += 30

    # Show the frame
    cv2.imshow("Traffic Vehicle Detection and Counting", frame)

    # Press ESC to stop the video
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Video processing complete.")
print("Red to Blue counts:", dict(count_red_to_blue))
print("Blue to Red counts:", dict(count_blue_to_red))
