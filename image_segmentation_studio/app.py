"""Browser interface for the Image Segmentation Studio.

Usage:
    python app.py        # then open http://127.0.0.1:5050
"""
import base64
import os

import matplotlib

matplotlib.use("Agg")  # the K-means module imports pyplot; no windows are needed here

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from src import utils
from src.kmeans_segmentation import segment_kmeans
from src.thresholding_segmentation import manual_thresholding, otsu_thresholding
from src.watershed_segmentation import segment_watershed

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "tests", "test_images")
MAX_SIDE = 800
MAX_UPLOADS = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
uploads = {}  # id -> (display name, BGR image), kept in memory only


def sample_names():
    return [os.path.basename(p) for p in utils.list_images(SAMPLE_DIR)]


def get_image(image_id):
    """Return a BGR image by sample file name or upload id, or None if unknown."""
    if image_id in uploads:
        return uploads[image_id][1]
    if image_id in sample_names():
        return utils.load_image(os.path.join(SAMPLE_DIR, image_id), MAX_SIDE)
    return None


def number(params, key, default, low, high, cast=float):
    """Read a numeric option, falling back to the default and limiting its range."""
    try:
        value = cast(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the image.")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def run_kmeans(image, params):
    k = number(params, "k", 3, 2, 12, int)
    return f"K-means (K={k})", segment_kmeans(image, k), None


def run_threshold(image, params):
    blur = bool(params.get("blur", False))
    blur_size = number(params, "blur_size", 5, 3, 31, int) | 1  # kernels must be odd

    if params.get("mode") == "manual":
        threshold = number(params, "threshold", 127, 0, 255, int)
        source = cv2.GaussianBlur(image, (blur_size, blur_size), 0) if blur else image
        _, result = manual_thresholding(source, threshold)
        name = "Manual threshold"
    else:
        threshold, result = otsu_thresholding(image, blur, blur_size)
        name = "Otsu threshold"

    if blur:
        name += f" + blur ({blur_size}x{blur_size})"
    return name, result, f"Threshold T = {int(threshold)}"


def run_watershed(image, params):
    labels = segment_watershed(
        image,
        colour_radius=number(params, "colour_radius", 30, 10, 60),
        peak_ratio=number(params, "peak_ratio", 0.6, 0.3, 0.9),
        min_region_fraction=number(params, "min_region", 0.4, 0, 3) / 100,
    )
    return "Watershed", labels, None


METHODS = {"kmeans": run_kmeans, "threshold": run_threshold, "watershed": run_watershed}


@app.route("/")
def index():
    return render_template("index.html", samples=sample_names())


@app.route("/samples/<path:name>")
def sample_file(name):
    if name not in sample_names():
        abort(404)
    return send_from_directory(SAMPLE_DIR, name)


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image was sent."), 400

    data = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        return jsonify(error="That file could not be read as an image."), 400

    if len(uploads) >= MAX_UPLOADS:
        uploads.pop(next(iter(uploads)))
    image_id = f"upload:{os.urandom(4).hex()}"
    name = os.path.basename(file.filename or "Uploaded image")
    uploads[image_id] = (name, utils.limit_size(image, MAX_SIDE))
    return jsonify(id=image_id, name=name, thumb=encode_png(utils.limit_size(image, 200)))


@app.route("/api/segment", methods=["POST"])
def segment():
    payload = request.get_json(silent=True) or {}
    image = get_image(payload.get("image"))
    if image is None:
        return jsonify(error="Unknown image."), 404

    method = payload.get("method")
    if method != "compare" and method not in METHODS:
        return jsonify(error="Unknown method."), 400

    params = payload.get("params")
    params = params if isinstance(params, dict) else {}
    results = []
    for key in (METHODS if method == "compare" else [method]):
        (name, raw, detail), seconds = utils.time_call(METHODS[key], image, params)
        results.append({
            "method": key,
            "name": name,
            "image": encode_png(utils.to_bgr(raw)),
            "ms": round(seconds * 1000, 2),
            "regions": utils.count_regions(raw),
            "detail": detail,
        })
    return jsonify(original=encode_png(image), results=results)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="That file is larger than the 16 MB limit."), 413


if __name__ == "__main__":
    print("Image Segmentation Studio: open http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050)
