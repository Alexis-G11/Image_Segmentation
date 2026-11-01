"""Tests for the browser interface."""
import base64
import io
import os

import cv2
import numpy as np
import pytest

from app import app, uploads
from src import utils
from src.watershed_segmentation import segment_watershed

TEST_IMAGES = os.path.join(os.path.dirname(__file__), "test_images")


@pytest.fixture
def client():
    uploads.clear()
    return app.test_client()


def segment(client, method, image="colored_balls.jpg", **params):
    return client.post("/api/segment", json={"image": image, "method": method, "params": params})


def png_upload(client):
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    _, data = cv2.imencode(".png", image)
    return client.post("/api/upload", data={"image": (io.BytesIO(data.tobytes()), "test.png")})


def decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_page_lists_shared_images(client):
    page = client.get("/").get_data(as_text=True)

    for name in ("apple.jpg", "colored_balls.jpg", "orange.jpg"):
        assert name in page


@pytest.mark.parametrize("method", ["kmeans", "threshold", "watershed"])
def test_each_method_returns_a_same_size_image(client, method):
    data = segment(client, method).get_json()

    assert len(data["results"]) == 1
    assert decode(data["results"][0]["image"]).shape == decode(data["original"]).shape


def test_compare_runs_all_three_methods(client):
    data = segment(client, "compare").get_json()

    assert [r["method"] for r in data["results"]] == ["kmeans", "threshold", "watershed"]


def test_k_changes_the_number_of_regions(client):
    data = segment(client, "kmeans", k=5).get_json()

    assert data["results"][0]["regions"] == 5
    assert data["results"][0]["name"] == "K-means (K=5)"


def test_otsu_reports_its_automatic_threshold(client):
    result = segment(client, "threshold", mode="otsu").get_json()["results"][0]

    assert result["detail"] == "Threshold T = 87"


def test_manual_threshold_uses_the_chosen_value(client):
    result = segment(client, "threshold", mode="manual", threshold=200).get_json()["results"][0]

    assert result["name"] == "Manual threshold"
    assert result["detail"] == "Threshold T = 200"


def test_blur_size_is_made_odd(client):
    result = segment(client, "threshold", blur=True, blur_size=8).get_json()["results"][0]

    assert "blur (9x9)" in result["name"]


def test_watershed_defaults_match_the_module(client):
    image = cv2.imread(os.path.join(TEST_IMAGES, "colored_balls.jpg"))
    expected = utils.count_regions(segment_watershed(image))

    result = segment(client, "watershed").get_json()["results"][0]

    assert result["regions"] == expected


def test_removing_small_regions_gives_fewer_regions(client):
    keep_all = segment(client, "watershed", min_region=0).get_json()["results"][0]
    remove_more = segment(client, "watershed", min_region=2).get_json()["results"][0]

    assert remove_more["regions"] < keep_all["regions"]


def test_out_of_range_options_are_limited(client):
    result = segment(client, "kmeans", k=999).get_json()["results"][0]

    assert result["regions"] == 12


def test_invalid_option_falls_back_to_default(client):
    result = segment(client, "kmeans", k="not a number").get_json()["results"][0]

    assert result["regions"] == 3


def test_unknown_image_is_rejected(client):
    assert segment(client, "kmeans", image="../../secret.jpg").status_code == 404


def test_unknown_method_is_rejected(client):
    assert segment(client, "magic").status_code == 400


def test_sample_route_rejects_other_files(client):
    assert client.get("/samples/..%2F..%2Fapp.py").status_code == 404


def test_uploaded_image_can_be_segmented(client):
    image_id = png_upload(client).get_json()["id"]

    response = segment(client, "kmeans", image=image_id, k=2)

    assert response.status_code == 200


def test_non_image_upload_is_rejected(client):
    response = client.post("/api/upload", data={"image": (io.BytesIO(b"not an image"), "x.txt")})

    assert response.status_code == 400
