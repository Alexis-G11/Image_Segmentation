import cv2
import numpy as np
import pytest

from src.thresholding_segmentation import (
    otsu_thresholding,
    manual_thresholding
)


def test_otsu_returns_binary_image():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 255

    threshold, result = otsu_thresholding(image)

    assert result.shape == image.shape
    assert set(np.unique(result)).issubset({0, 255})
    assert 0 <= threshold <= 255


def test_otsu_accepts_colour_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = (255, 255, 255)

    threshold, result = otsu_thresholding(image)

    assert result.shape == image.shape[:2]
    assert 0 <= threshold <= 255


def test_otsu_with_gaussian_blur():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 50:] = 255

    threshold, result = otsu_thresholding(image, apply_blur=True)

    assert result.shape == image.shape
    assert set(np.unique(result)).issubset({0, 255})


def test_manual_thresholding():
    image = np.array([[50, 100, 150, 200]], dtype=np.uint8)

    threshold, result = manual_thresholding(image, threshold=127)

    expected = np.array([[0, 0, 255, 255]], dtype=np.uint8)

    assert threshold == 127
    assert np.array_equal(result, expected)


def test_manual_threshold_rejects_invalid_value():
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        manual_thresholding(image, threshold=300)


def test_otsu_rejects_empty_image():
    with pytest.raises(ValueError):
        otsu_thresholding(None)