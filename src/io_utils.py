"""
Image loading and saving helpers.

Keeps file I/O out of the preprocessing and extraction modules
so those can focus on image transformations only.
"""

from pathlib import Path

import cv2


def load_image(path: str):
    """
    Load an image from disk.

    Returns a BGR numpy array (OpenCV default).
    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the file exists but is not a readable image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No image at: {path}")

    image = cv2.imread(str(p))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    return image


def save_image(image, path: str) -> None:
    """Write an image to disk. Creates parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def to_grayscale(image):
    """Convert a BGR image to grayscale. Passes through if already 2D."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
