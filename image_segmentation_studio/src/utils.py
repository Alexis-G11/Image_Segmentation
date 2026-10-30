"""Shared helpers for the Image Segmentation Studio (Member 4).

Agreed function interface for every segmentation module:

    segment_xxx(image_bgr: np.ndarray, **params) -> np.ndarray
        image_bgr : uint8 image, shape (H, W, 3), BGR order (OpenCV)
        returns   : uint8 image, same H x W, either
                    - (H, W, 3) BGR visualisation of the segmentation, or
                    - (H, W) single-channel mask (0/255 or labels)

The studio converts whatever comes back into a displayable BGR image,
so members only need to respect H x W and uint8 / label values.
"""
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------- I/O
def load_image(path, max_side=None):
    """Load a BGR image. Optionally downscale so the longest side <= max_side."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if max_side:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    return img


def save_image(path, image):
    """Save an image, creating the parent folder if needed."""
    ensure_dir(os.path.dirname(path) or ".")
    if not cv2.imwrite(path, image):
        raise IOError(f"Cannot write image: {path}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_images(folder):
    """Sorted list of image paths inside a folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTENSIONS))


# ------------------------------------------------------------- display
def to_bgr(result, reference=None):
    """Convert a method output to a 3-channel uint8 BGR image."""
    arr = np.asarray(result)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported result shape: {arr.shape}")
    if arr.dtype != np.uint8:  # label map (e.g. watershed markers)
        return colorize_labels(arr)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def colorize_labels(labels):
    """Give each label a stable random colour; -1 (boundary) is drawn red."""
    labels = np.asarray(labels).astype(np.int32)
    rng = np.random.RandomState(42)
    n = max(int(labels.max()) + 1, 2)
    palette = rng.randint(40, 255, size=(n, 3)).astype(np.uint8)
    out = palette[np.clip(labels, 0, n - 1)]
    out[labels == -1] = (0, 0, 255)
    return out


def count_regions(result):
    """Rough number of distinct regions/colours in a result (for the table)."""
    arr = np.asarray(result)
    n = len(np.unique(arr.reshape(-1, arr.shape[2]), axis=0)) if arr.ndim == 3 \
        else len(np.unique(arr))
    # Photos with overlaid contours have thousands of colours: not a region count.
    return int(n) if n <= 64 else 0  # 0 = not meaningful


# -------------------------------------------------------------- timing
@contextmanager
def timer():
    """with timer() as t: ...; then t['seconds']."""
    box = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start


def time_call(func, *args, repeats=1, **kwargs):
    """Run func `repeats` times, return (last_result, mean_seconds)."""
    result, total = None, 0.0
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        total += time.perf_counter() - start
    return result, total / max(1, repeats)


# --------------------------------------------------------------- mocks
# Temporary stand-ins used until teammates push their real algorithms.
def mock_kmeans(image, k=3):
    small = cv2.resize(image, (image.shape[1] // 4 or 1, image.shape[0] // 4 or 1))
    out = (small // 85) * 85  # crude colour quantisation
    return cv2.resize(out, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def mock_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]


def mock_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    out = image.copy()
    out[edges > 0] = (0, 0, 255)
    return out