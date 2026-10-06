"""
Basic tests for the preprocessing pipeline.

Run with: pytest tests/
"""

import numpy as np

from src.preprocess import to_grayscale, threshold, preprocess_pipeline


def test_to_grayscale_on_bgr():
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = to_grayscale(bgr)
    assert gray.shape == (10, 10)


def test_to_grayscale_passthrough():
    gray_in = np.zeros((10, 10), dtype=np.uint8)
    gray_out = to_grayscale(gray_in)
    assert gray_out.shape == (10, 10)


def test_threshold_rejects_even_block_size():
    img = np.zeros((10, 10), dtype=np.uint8)
    try:
        threshold(img, block_size=30)
    except ValueError:
        return
    assert False, "Expected ValueError for even block_size"


def test_pipeline_output_is_binary():
    # Random grayscale image, pipeline should return 0 or 255 only
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    out = preprocess_pipeline(img, denoise_strength=0)
    unique = set(np.unique(out).tolist())
    assert unique.issubset({0, 255})
