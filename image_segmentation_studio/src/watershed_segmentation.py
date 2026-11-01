import cv2
import numpy as np

# Values chosen by comparing results on the three shared test images.
SPATIAL_RADIUS = 15     # mean shift: how far to look for similar pixels
COLOUR_RADIUS = 30      # mean shift: how different a colour may be and still blend
MEDIAN_SIZE = 5         # removes the small speckles mean shift leaves behind
PEAK_WINDOW = 51        # size of the neighbourhood used to find region centres
PEAK_RATIO = 0.6        # a seed must be this close to the local peak
MIN_EDGE_DISTANCE = 2.0  # a seed must be at least this far from an edge (pixels)
MIN_REGION_FRACTION = 0.004  # regions smaller than this share of the image are removed


def smooth_image(image, colour_radius=COLOUR_RADIUS):
    """Flatten texture and small highlights while keeping object edges sharp."""
    smooth = cv2.pyrMeanShiftFiltering(image, SPATIAL_RADIUS, colour_radius)
    return cv2.medianBlur(smooth, MEDIAN_SIZE)


def find_edges(smooth):
    """Return a binary edge map, using colour differences as well as brightness."""
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)

    grad_x = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt((grad_x ** 2 + grad_y ** 2).sum(axis=2))

    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(
        gradient.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return edges


def find_markers(edges, peak_ratio=PEAK_RATIO):
    """
    Place one starting marker near the centre of each area enclosed by edges.

    The distance transform is highest far from any edge, so each peak is the
    middle of a region. This separates touching objects without needing a
    foreground mask, which a single brightness threshold cannot give for
    images containing many colours.
    """
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    window = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PEAK_WINDOW, PEAK_WINDOW)
    )
    local_peak = cv2.dilate(distance, window)
    seeds = (distance >= peak_ratio * local_peak) & (distance > MIN_EDGE_DISTANCE)

    _, markers = cv2.connectedComponents(seeds.astype(np.uint8))

    return markers.astype(np.int32)


def remove_small_markers(labels, markers, min_area):
    """Drop markers whose region is tiny, such as a highlight on a shiny object."""
    region_ids, areas = np.unique(labels[labels > 0], return_counts=True)
    keep = region_ids[areas >= min_area]

    if keep.size == 0:
        keep = region_ids[np.argsort(areas)[-1:]]

    return np.where(np.isin(markers, keep), markers, 0).astype(np.int32)


def segment_watershed(image, colour_radius=COLOUR_RADIUS, peak_ratio=PEAK_RATIO,
                      min_region_fraction=MIN_REGION_FRACTION):
    """
    Apply marker-based watershed segmentation.

    Args:
        image: Input BGR colour image (uint8).
        colour_radius: Higher values smooth more, so fewer edges remain.
        peak_ratio: Higher values make markers smaller, which splits regions more.
        min_region_fraction: Regions below this share of the image are removed.

    Returns:
        labels: int32 label map of the same height and width. Each region has
        its own positive number and boundary pixels are -1.
    """
    if image is None:
        raise ValueError("Input image is empty or could not be loaded.")

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Input must be a uint8 BGR colour image.")

    smooth = smooth_image(image, colour_radius)
    markers = find_markers(find_edges(smooth), peak_ratio)

    # Without any marker there is nothing to flood from: one region only.
    if markers.max() == 0:
        return np.ones(image.shape[:2], dtype=np.int32)

    # First flood finds the regions, second floods again without the tiny ones.
    labels = cv2.watershed(smooth, markers.copy())
    min_area = min_region_fraction * image.shape[0] * image.shape[1]
    markers = remove_small_markers(labels, markers, min_area)

    return cv2.watershed(smooth, markers)
