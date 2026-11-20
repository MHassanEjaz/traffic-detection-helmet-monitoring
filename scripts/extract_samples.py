
import cv2
from pathlib import Path

VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("sample_frames")

OUTPUT_DIR.mkdir(exist_ok=True)

for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_folder = OUTPUT_DIR / video_path.stem
    output_folder.mkdir(exist_ok=True)

    sample_positions = [
        int(total_frames * ratio)
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    for i, frame_number in enumerate(sample_positions, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if success:
            output_path = output_folder / f"sample_{i}.jpg"
            cv2.imwrite(str(output_path), frame)

    cap.release()
    print(f"Extracted samples: {video_path.name}")

print("\nDone! Check sample_frames folder.")
