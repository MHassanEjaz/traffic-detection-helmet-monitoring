
import cv2
from pathlib import Path

VIDEO_DIR = Path("test_videos")

for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"\nCould not open: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else 0

    print(f"\nVideo: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {frames}")
    print(f"Duration: {duration:.2f} seconds")

    cap.release()
