"""Live demo script - Image Segmentation Studio.

Usage:
    python demo.py                      # all images in tests/test_images
    python demo.py --image photo.jpg    # one image
    python demo.py --folder my_images --no-show
    python demo.py --synthetic          # built-in test image (no files needed)
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from segmentation_studio import ImageSegmentationStudio  # noqa: E402
import utils  # noqa: E402


def synthetic_image(size=400):
    """Touching coloured discs on uneven background (shows failure cases)."""
    rng = np.random.RandomState(0)
    img = np.full((size, size, 3), 230, np.uint8)
    grad = np.linspace(0, 60, size).astype(np.uint8)
    img = cv2.subtract(img, np.tile(grad, (size, 1))[:, :, None].repeat(3, 2))
    for cx, cy, r, col in [(130, 140, 60, (60, 60, 200)), (200, 150, 55, (60, 180, 60)),
                           (260, 260, 50, (200, 90, 40)), (120, 280, 45, (30, 160, 220))]:
        cv2.circle(img, (cx, cy), r, col, -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Image Segmentation Studio demo")
    p.add_argument("--image", help="path to a single image")
    p.add_argument("--folder", default=os.path.join(HERE, "tests", "test_images"),
                   help="folder of images (default: tests/test_images)")
    p.add_argument("--output", default=os.path.join(HERE, "..", "results"))
    p.add_argument("--max-side", type=int, default=800)
    p.add_argument("--repeats", type=int, default=3, help="timing repetitions")
    p.add_argument("--synthetic", action="store_true", help="use a generated image")
    p.add_argument("--no-show", action="store_true", help="save only, no window")
    args = p.parse_args()

    studio = ImageSegmentationStudio(args.output, args.max_side, args.repeats)
    print("Method status:", studio.status())
    show = not args.no_show

    folder_has_images = bool(utils.list_images(args.folder))
    if not args.image and not args.synthetic and not folder_has_images:
        print(f"[!] No images found in {os.path.abspath(args.folder)}\n"
              "    (Member 1 has not uploaded them yet?) "
              "-> using the built-in synthetic image instead.\n")
        args.synthetic = True

    if args.synthetic:
        path = os.path.join(studio.output_dir, "synthetic.png")
        utils.save_image(path, synthetic_image())
        _, res = studio.process_file(path, show=show)
        studio.save_table({"synthetic.png": res})
        all_results = {"synthetic.png": res}
    elif args.image:
        _, res = studio.process_file(args.image, show=show)
        all_results = {os.path.basename(args.image): res}
        studio.save_table(all_results)
    else:
        all_results = studio.process_folder(args.folder)

    print("\n" + studio.table_text(all_results))
    print(f"\nOutputs saved in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()