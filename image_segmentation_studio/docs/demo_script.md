## Full Project Demonstration

Before presenting, open a terminal in `image_segmentation_studio/`, activate the environment and start the interface:

```bash
python app.py
```

Then open <http://127.0.0.1:5050> in a browser, and leave both running. The audience only needs to see the browser. No internet connection is needed.

1. **K-means:** with the coloured balls selected, move K from 3 to 8. More colours are kept, but the balls are still not separate objects.
2. **Thresholding:** select Thresholding. Otsu chooses T = 87 by itself. Switch to Manual and move T to show how sensitive the result is, then turn on blur.
3. **Watershed:** select Watershed. Most balls get their own region (43 regions). Raise the smallest region slider to 3% and the count drops to 12, then press Reset options.
4. **Other images:** select the orange to show watershed's cleanest result: 3 regions.
5. **Compare all:** select the coloured balls and Compare all, then point out the times under each result.
6. Finish with the conclusion: no method is best on every image.

If the interface fails, show slide 11 or the saved figures in `results/` instead.

## Watershed Specialist Demonstration

1. Explain that edges act as ridges, and that water rises from markers.
2. Select Watershed in the interface with the orange: 3 clean regions.
3. Select the coloured balls and show that touching balls get separate regions.
4. Move each option and explain its effect: smoothing, marker strictness and smallest region.
5. Select the apple to show the limitation: its streaks split the upper left into extra regions.

## Testing Demonstration

1. Explain why the three shared images were chosen.
2. Run the tests in a terminal if the lecturer asks to see them: `python -m pytest tests -q`.
3. Explain what the tests check, using `docs/test_plan.md`.
4. State the limitation: accuracy was judged by eye, because the images have no hand-labelled answers.

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
