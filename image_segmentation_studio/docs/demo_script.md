## Full Project Demonstration

Before starting, open a terminal in `image_segmentation_studio/`, activate the environment and keep `results/` open as a backup.

1. Run the studio on the coloured balls:

   ```bash
   python demo.py --image tests/test_images/colored_balls.jpg
   ```

2. Show the status line in the terminal: every method says `REAL`.
3. Show the comparison figure and explain each column:
   - K-means: only 3 colours, so warm-coloured balls merge.
   - Otsu: brightness only, so dark balls disappear into the gaps.
   - Otsu + blur: same threshold, slightly smoother shapes.
   - Watershed: most balls get their own region, even when touching.
4. Run all three images:

   ```bash
   python demo.py --no-show
   ```

5. Show the printed performance table: Otsu takes under 1 ms, while K-means and watershed take from about 100 ms to about 500 ms.
6. Open `results/orange/orange_comparison.png` to show watershed's cleanest result.
7. Finish with the conclusion: no method is best on every image.

If the live run fails, show the saved figures in `results/` instead.

## Thresholding Specialist Demonstration

Run from `image_segmentation_studio/` with `python demo_thresholding.py`.

1. Open the project and show the thresholding implementation.
2. Explain that the input image is converted to grayscale.
3. Run Otsu thresholding without Gaussian blur.
4. Run Otsu thresholding with Gaussian blur.
5. Run manual thresholding using threshold 127.
6. Show the three output images for the apple.
7. Repeat the comparison for the coloured ball and orange.
8. Explain the differences in the output images.
9. Discuss whether Gaussian blur improved the segmentation.
10. Explain the limitations of thresholding when foreground and background intensities overlap.

The demonstration produces nine output images from the three shared input images.
