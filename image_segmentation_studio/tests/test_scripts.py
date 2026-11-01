"""Tests for the watershed module, the shared interface and the studio."""
import os

import cv2
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src import utils
from src.kmeans_segmentation import segment_kmeans
from src.segmentation_studio import ImageSegmentationStudio
from src.thresholding_segmentation import segment_otsu, segment_otsu_blur
from src.watershed_segmentation import segment_watershed

TEST_IMAGES = os.path.join(os.path.dirname(__file__), "test_images")
SHARED_IMAGES = ["apple.jpg", "colored_balls.jpg", "orange.jpg"]
ALL_METHODS = [segment_kmeans, segment_otsu, segment_otsu_blur, segment_watershed]


def touching_discs():
    """Two same-coloured discs that touch: the classic watershed case."""
    image = np.full((200, 300, 3), 240, dtype=np.uint8)
    cv2.circle(image, (100, 100), 60, (40, 40, 200), -1)
    cv2.circle(image, (220, 100), 60, (40, 40, 200), -1)
    return image


def label_at(labels, x, y):
    return labels[y, x]


# ------------------------------------------------------- shared interface
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("name", SHARED_IMAGES)
def test_method_keeps_image_size(method, name):
    image = cv2.imread(os.path.join(TEST_IMAGES, name))

    result = method(image)

    assert result.shape[:2] == image.shape[:2]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_output_can_be_displayed(method):
    result = method(touching_discs())

    display = utils.to_bgr(result)

    assert display.shape == (200, 300, 3)
    assert display.dtype == np.uint8


def test_kmeans_uses_k_colours():
    result = segment_kmeans(touching_discs(), k=2)

    assert utils.count_regions(result) == 2


def test_otsu_is_binary():
    result = segment_otsu(touching_discs())

    assert set(np.unique(result)).issubset({0, 255})


# -------------------------------------------------------------- watershed
def test_watershed_separates_touching_discs():
    labels = segment_watershed(touching_discs())

    left = label_at(labels, 100, 100)
    right = label_at(labels, 220, 100)
    background = label_at(labels, 10, 10)

    assert left > 0 and right > 0 and background > 0
    assert len({left, right, background}) == 3


def test_watershed_finds_one_region_in_plain_image():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    labels = segment_watershed(image)

    assert utils.count_regions(labels) == 1


@pytest.mark.parametrize("bad_input", [None, np.zeros((50, 50), dtype=np.uint8)])
def test_watershed_rejects_invalid_input(bad_input):
    with pytest.raises(ValueError):
        segment_watershed(bad_input)


# ------------------------------------------------------------------ utils
def test_count_regions_ignores_watershed_boundaries():
    labels = np.array([[1, -1, 2], [1, -1, 2]], dtype=np.int32)

    assert utils.count_regions(labels) == 2


def test_load_image_downscales_long_side():
    image = utils.load_image(os.path.join(TEST_IMAGES, "apple.jpg"), max_side=100)

    assert max(image.shape[:2]) == 100


def test_load_image_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        utils.load_image("does_not_exist.jpg")


# ----------------------------------------------------------------- studio
def test_studio_uses_only_real_methods(tmp_path):
    studio = ImageSegmentationStudio(str(tmp_path), repeats=1)

    assert set(studio.status().values()) == {"REAL"}


def test_studio_survives_a_failing_method(tmp_path):
    def broken(image):
        raise RuntimeError("broken method")

    studio = ImageSegmentationStudio(str(tmp_path), repeats=1)
    studio.register("Broken", broken)

    results = studio.segment(touching_discs())

    assert results["Broken"]["error"] == "broken method"
    assert results["K-means (K=3)"]["error"] is None


def test_studio_saves_figures_and_tables(tmp_path):
    studio = ImageSegmentationStudio(str(tmp_path), repeats=1)

    results = studio.process_folder(TEST_IMAGES)

    assert sorted(results) == SHARED_IMAGES
    assert (tmp_path / "performance.csv").is_file()
    assert (tmp_path / "performance.md").is_file()
    for stem in ("apple", "colored_balls", "orange"):
        for tag in ("comparison", "kmeans", "otsu", "otsu_blur", "watershed"):
            assert (tmp_path / stem / f"{stem}_{tag}.png").is_file()
