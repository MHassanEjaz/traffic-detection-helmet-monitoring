
import cv2
from pathlib import Path

VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("datasets/raw_frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"Invalid FPS: {video_path.name}")
        cap.release()
        continue

    video_output = OUTPUT_DIR / video_path.stem
    video_output.mkdir(parents=True, exist_ok=True)

    interval = round(fps)
    saved = 0

    for frame_number in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            continue

        filename = f"frame_{saved + 1:04d}.jpg"
        cv2.imwrite(str(video_output / filename), frame)
        saved += 1

    cap.release()
    print(f"{video_path.name}: {saved} frames extracted")

print("\nExtraction complete!")
