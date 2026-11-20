
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv(ROOT / ".env")


def get_path(variable_name, default_path):
    value = os.getenv(variable_name, default_path)
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path


# Model paths
MODEL_PATH = get_path("MODEL_PATH", "models/yolo26n.pt")
TRACKER_PATH = get_path(
    "TRACKER_PATH",
    "trackers/bytetrack_custom.yaml"
)

HELMET_MODELS = {
    "Original Helmet Model": get_path(
        "ORIGINAL_HELMET_MODEL", "models/helmet.pt"
    ),
    "Fine-tuned Helmet Model": get_path(
        "FINETUNED_HELMET_MODEL", "models/helmet_finetuned.pt"
    ),
    "RideSafe Helmet Model": get_path(
        "RIDESAFE_HELMET_MODEL", "models/helmet_ridesafe_best.pt"
    )
}

# Vehicle detection settings
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.15")
)
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.50"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# Helmet detection settings
HELMET_CONFIDENCE = float(
    os.getenv("HELMET_CONFIDENCE", "0.10")
)
HELMET_IMAGE_SIZE = int(
    os.getenv("HELMET_IMAGE_SIZE", "1280")
)
HELMET_DETECTION_INTERVAL = int(
    os.getenv("HELMET_DETECTION_INTERVAL", "3")
)
HELMET_MAX_AGE = int(os.getenv("HELMET_MAX_AGE", "2"))

# Counting line
LINE_POSITION_RATIO = float(
    os.getenv("LINE_POSITION_RATIO", "0.50")
)
