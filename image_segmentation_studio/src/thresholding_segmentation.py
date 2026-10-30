import cv2
import numpy as np


def convert_to_grayscale(image):
    """Convert a colour image to grayscale or copy an existing grayscale image."""
    if image is None:
        raise ValueError("Input image is empty or could not be loaded.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Input must be a grayscale or colour image.")

    if gray.dtype != np.uint8:
        raise ValueError("Input image must use uint8 pixel values.")

    return gray


def otsu_thresholding(image, apply_blur=False):
    """
    Apply Otsu's automatic thresholding.

    Args:
        image: Input colour or grayscale image.
        apply_blur: Apply Gaussian blur before thresholding if True.

    Returns:
        threshold_value: Automatically selected threshold.
        segmented_image: Binary image containing 0 and 255 values.
    """
    gray = convert_to_grayscale(image)

    if apply_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_value, segmented_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold_value, segmented_image


def manual_thresholding(image, threshold=127):
    """
    Apply binary thresholding using a manually selected threshold.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number.")

    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")

    gray = convert_to_grayscale(image)

    _, segmented_image = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return threshold, segmented_image