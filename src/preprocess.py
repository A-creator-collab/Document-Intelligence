"""
Image preprocessing for OCR.

Takes a raw image and returns a cleaned image ready for text extraction.
The pipeline is intentionally simple: grayscale → denoise → threshold → deskew.

Each step is exposed as its own function so they can be tested and reordered.
"""

import cv2
import numpy as np


def to_grayscale(image):
    """Convert to grayscale. Passes through if already single channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image, strength: int = 10):
    """
    Remove noise while preserving text edges.

    Uses fastNlMeansDenoising, which works well on scanned documents.
    Higher `strength` removes more noise but can blur thin strokes.
    """
    if strength <= 0:
        return image
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def threshold(image, block_size: int = 31, c: int = 15):
    """
    Binarize the image using adaptive Gaussian thresholding.

    Adaptive thresholding handles uneven lighting across a scan
    better than a single global threshold.

    block_size must be odd and > 1. c is subtracted from the local mean.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")
    if block_size <= 1:
        raise ValueError("block_size must be > 1")

    return cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )


def deskew(image, max_angle: float = 15.0):
    """
    Detect and correct small rotations in a scanned document.

    Only corrects angles within ±max_angle. Larger angles are usually
    not skew — they're the document being rotated on purpose.
    """
    inverted = cv2.bitwise_not(image)
    coords = np.column_stack(np.where(inverted > 0))
    if coords.shape[0] < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_pipeline(image, denoise_strength: int = 10):
    """
    Run the full preprocessing chain on a BGR or grayscale image.

    Returns a single-channel binarized, deskewed image ready for OCR.
    Order matters: denoise before threshold so noise doesn't get binarized in.
    Deskew runs last on the clean binary image for a stable angle estimate.
    """
    gray = to_grayscale(image)
    cleaned = denoise(gray, strength=denoise_strength)
    binary = threshold(cleaned)
    straightened = deskew(binary)
    return straightened
