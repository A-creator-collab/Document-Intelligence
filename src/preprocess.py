"""
Image preprocessing for OCR.

Functions here take a raw image and return a cleaned image
ready for text extraction (grayscale, denoise, threshold, deskew).
"""

from pathlib import Path


def load_image(path: str):
    """Load an image from disk. Returns a numpy array."""
    raise NotImplementedError("Install opencv-python and implement this.")


def to_grayscale(image):
    """Convert a BGR image to grayscale."""
    raise NotImplementedError


def denoise(image):
    """Remove noise while preserving edges."""
    raise NotImplementedError


def threshold(image):
    """Apply binarization to separate text from background."""
    raise NotImplementedError


def deskew(image):
    """Detect and correct image rotation."""
    raise NotImplementedError


def preprocess_pipeline(path: str):
    """Run the full preprocessing chain on an image path."""
    raise NotImplementedError


if __name__ == "__main__":
    print("Preprocessing module — not wired up yet.")
