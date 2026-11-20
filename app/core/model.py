
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO


@lru_cache(maxsize=8)
def load_model(model_path: str) -> YOLO:
    """
    Load and cache a YOLO model.

    Args:
        model_path: Path to the YOLO model weights.

    Returns:
        Loaded YOLO model.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return YOLO(str(path))
