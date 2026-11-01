# Test Plan

## Shared dataset

All methods use the images in `tests/test_images/`.

| Image | Size (pixels) | What it tests |
|---|---|---|
| `apple.jpg` | 505 x 501 | One object with skin streaks, a highlight, a leaf and a shadow |
| `orange.jpg` | 498 x 501 | One object with rough texture, a leaf and a soft shadow |
| `colored_balls.jpg` | 505 x 502 | Many touching objects in many colours, with highlights |

## Expected behaviour

| Method | Input | Expected output | Expected to struggle with |
|---|---|---|---|
| K-means | BGR image, K | Same-size BGR image with exactly K colours | Different objects of similar colour; choosing K |
| Otsu | BGR or grayscale image | Same-size image holding only 0 and 255, with an automatic threshold | Shadows, dark objects, overlapping brightness |
| Otsu + blur | As Otsu | As Otsu, with smoother boundaries and fewer specks | Fine details can be lost |
| Manual threshold | Image, T from 0 to 255 | Same-size image holding only 0 and 255 | A single T does not suit every image |
| Watershed | BGR image | Same-size label map, with -1 on boundaries | Too many regions; overlapping objects with no visible edge |

## Test cases

Run all tests from `image_segmentation_studio/`:

```bash
python -m pytest tests -q
```

| File | Covers |
|---|---|
| `tests/test_thresholding.py` | Binary output, colour input, blur, manual thresholds, invalid input |
| `tests/test_scripts.py` | Output size and display on all images, K-means colours, watershed separation and invalid input, helpers, studio behaviour and saved files |
| `tests/test_app.py` | Browser interface: pages, every method, adjustable options, uploads and bad requests |

## What is not tested

- **Accuracy:** the images have no hand-labelled correct answers, so segmentation quality is judged by eye.
- **Exact times:** times change on every run, so tests never check them.
- **Browser display:** automated tests check the interface's responses, not how the page looks.
