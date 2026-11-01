from pathlib import Path

import cv2

from src.thresholding_segmentation import (
    otsu_thresholding,
    manual_thresholding
)


IMAGE_DIR = Path("tests/test_images")
RESULT_DIR = Path("tests/test_results/thresholding")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {IMAGE_DIR.resolve()}")
        print("Check that your team's images are in tests/test_images.")
        return

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        # Method 1: Otsu without Gaussian blur
        threshold_plain, result_plain = otsu_thresholding(
            image, apply_blur=False
        )

        # Method 2: Otsu with Gaussian blur
        threshold_blur, result_blur = otsu_thresholding(
            image, apply_blur=True
        )

        # Method 3: Manual threshold
        threshold_manual, result_manual = manual_thresholding(
            image, threshold=127
        )

        # Save the three results
        name = image_path.stem

        cv2.imwrite(
            str(RESULT_DIR / f"{name}_otsu_no_blur.png"),
            result_plain
        )

        cv2.imwrite(
            str(RESULT_DIR / f"{name}_otsu_blur.png"),
            result_blur
        )

        cv2.imwrite(
            str(RESULT_DIR / f"{name}_manual_127.png"),
            result_manual
        )

        print(f"\nImage: {image_path.name}")
        print(f"Otsu without blur: {threshold_plain:.0f}")
        print(f"Otsu with blur:    {threshold_blur:.0f}")
        print(f"Manual threshold:  {threshold_manual}")

    print("\nThresholding demonstration completed.")
    print(f"Results saved in: {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    main()