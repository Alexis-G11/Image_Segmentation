## Thresholding Segmentation

### 1. Introduction

Thresholding is an image segmentation technique that separates an image into regions based on pixel intensity. It converts a grayscale image into a binary image, where pixels are classified as foreground or background.

In this project, Otsu's automatic thresholding is implemented and compared with manual thresholding. Gaussian blur is also applied as a preprocessing technique to investigate its effect on segmentation quality.

### 2. Theoretical Background

A grayscale image contains pixel intensity values ranging from 0 to 255. Thresholding uses a threshold value T to classify each pixel:

- If the pixel intensity is greater than T, it is assigned 255 (white).
- Otherwise, it is assigned 0 (black).

Otsu's method automatically determines a threshold by analysing the grayscale image histogram. It selects the threshold that maximises the between-class variance of the foreground and background classes. This is equivalent to minimising their weighted within-class variance.

Manual thresholding uses a threshold selected by the user. In this project, a threshold of 127 is used as a baseline for comparison.

### 3. Gaussian Blur Preprocessing

Gaussian blur is applied before thresholding to reduce noise and small intensity variations. This can make the segmentation more stable.

However, excessive blurring may remove small details and soften object boundaries. Therefore, the results with and without Gaussian blur are compared.

### 4. Implementation

The thresholding module implements two methods:

1. Otsu thresholding without preprocessing.
2. Otsu thresholding after Gaussian blur.
3. Manual thresholding using a fixed threshold of 127.

The implementation uses Python, OpenCV and NumPy. The input images are converted to grayscale before thresholding.

### 5. Experimental Design

The same three images used by the other project members are used for testing: apple, coloured ball and orange.

Each image is processed using the three configurations:

- Otsu without Gaussian blur.
- Otsu with Gaussian blur.
- Manual thresholding with T = 127.

This produces nine segmentation outputs in total.

### 6. Results and Discussion

The automatically selected Otsu threshold values and the resulting binary images are examined for each input image.

The comparison considers:
- Whether the foreground object is separated clearly from the background.
- Whether noise is present in the segmented output.
- Whether object boundaries and small details are preserved.
- Whether Gaussian blur improves or reduces segmentation quality.
- How the automatic threshold compares with the fixed manual threshold.

**Insert the actual threshold values and observations from the experiments here. Do not estimate or invent results.**

### 7. Limitations

Thresholding relies on grayscale intensity differences. If the foreground and background have similar grayscale intensities, they may not be separated correctly.

Otsu's method works best when the image histogram contains reasonably distinct intensity groups. Gaussian blur can reduce noise but may also remove fine details.

### 8. Conclusion

Otsu's method provides an automatic way to select a threshold without requiring the user to choose one manually. Comparing it with a fixed manual threshold helps demonstrate the advantages and limitations of automatic threshold selection.

The Gaussian blur experiment shows how preprocessing can influence the final segmentation result. The most suitable method depends on the image and the intensity differences between its foreground and background.