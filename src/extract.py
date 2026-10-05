"""
Text extraction from preprocessed document images.

This is where the OCR engine gets called (Tesseract, EasyOCR, etc.)
and where raw output gets structured into clean text.
"""

def extract_text(image):
    """Run OCR on a preprocessed image. Returns raw text."""
    raise NotImplementedError


def extract_with_boxes(image):
    """Run OCR and return text plus bounding boxes per word."""
    raise NotImplementedError


def clean_text(raw_text: str) -> str:
    """Fix common OCR artifacts: stray symbols, broken line breaks."""
    raise NotImplementedError
