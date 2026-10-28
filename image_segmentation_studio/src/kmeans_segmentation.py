
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path


def kmeans_segmentation(image_path, image_name):

    project_dir = Path(__file__).resolve().parent.parent
    input_path = Path(image_path)
    if not input_path.is_absolute() and not input_path.is_file():
        input_path = project_dir / "tests" / "test_images" / input_path

    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Convert BGR to RGB for display
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape pixels into a 2D array
    pixels = image.reshape((-1, 3))

    # Convert pixels to float32
    pixels = np.float32(pixels)

    # Define stopping criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Set K = 3
    K = 3

    # Apply K-means and measure time
    start = time.perf_counter()

    compactness, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    end = time.perf_counter()

    # Replace pixels with cluster colours
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]

    #  Reconstruct the image
    segmented = segmented.reshape(image.shape)

    # Convert result to RGB
    segmented_rgb = cv2.cvtColor(
        segmented, cv2.COLOR_BGR2RGB
    )

    #  Display original and segmented images
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original " + image_name)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title("K-means " + image_name + " (K=3)")
    plt.axis("off")

    plt.tight_layout()

    output_path = project_dir / f"kmeans_{input_path.name}"
    preview_path = project_dir / f"kmeans_{input_path.stem}_comparison.png"
    if not cv2.imwrite(str(output_path), segmented):
        raise OSError(f"Could not save result: {output_path}")
    plt.savefig(preview_path)
    plt.show()
    plt.close()

    print("\nImage:", image_name)
    print("Number of clusters:", K)
    print("Execution time:", round(end - start, 4), "seconds")
    print("Compactness:", compactness)
    print("Saved as:", output_path)
    print("Comparison preview:", preview_path)


# IMAGE 1: APPLE
kmeans_segmentation("apple.jpg", "Apple")

# IMAGE 2: ORANGE
kmeans_segmentation("orange.jpg", "Orange")

# IMAGE 3: COLOURED BALLS
kmeans_segmentation("colored_balls.jpg", "Coloured Balls")
