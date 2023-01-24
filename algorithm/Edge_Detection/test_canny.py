import canny_compare  # test file which changes what convolution is used based on type number
import gaussian_smoothing  # file with library function integrated into gaussian smoothing
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter


# Anvit Sinha 10/2/2022
# File to test the algorithm for canny edge detection

def main():

    time_convolve(plt.imread('100_baht_old.jpg', 0))


# Anvit Sinha 11/13/2022
# Function to test how the library function integrates with our methods
def library_convolve_test(img):
    img = canny_compare.grey_scale(img)

    old_gauss = canny_compare.gaussian_smoothing(img, 1)
    new_gauss = gaussian_smoothing.gaussian_smoothing(img, 1)

    plot_comparison(old_gauss, "Old Gaussian", new_gauss, "New Gaussian")


# Anvit Sinha 10/14/2022
# Function plots the original image and image after edge detection for 3 images selected to show
# working of the algorithm in various conditions
def canny_overall():
    # Anvit Sinha 11/13/2022
    # refactored testing of canny edge detector BEFORE using library functions
    # into a separate function

    # Test each function

    # Read images to test and make a copy to not modify original image

    # image of ideal note
    img1 = plt.imread('50_baht_test.jpg', 0)
    img1_copy = np.copy(img1)

    # image of note on a table
    img2 = plt.imread('100_baht_test.jpg', 0)
    img2_copy = np.copy(img2)

    img3 = plt.imread('100_baht_old.jpg', 0)
    img3_copy = np.copy(img3)

    originals = [img1, img2, img3]
    copies = [img1_copy, img2_copy, img3_copy]

    for o, c in zip(originals, copies):
        plot_comparison(o, "Original", canny_compare.edge_detection(c, 2), "After Edge Detection")


# Anvit Sinha 11/15/2022
# test the time difference between the old and new implementations
# of convolution by timing the functions that use convolution
def time_convolve(img1):
    img1_gray = canny_compare.grey_scale(img1)  # grey scale the image

    t1 = perf_counter()  # initial time

    gauss_old = canny_compare.gaussian_smoothing(img1_gray, 2, 1)    # using old convolution

    t2 = perf_counter()  # time after old gaussian

    gauss_new = canny_compare.gaussian_smoothing(img1_gray, 2, 0)    # using new convolution

    t3 = perf_counter()     # time after new gaussian

    gradient_old = canny_compare.gradient_calc(gauss_old, 1)[0]    # using old convolution

    t4 = perf_counter()     # time after old gradient calculations

    gradient_new = canny_compare.gradient_calc(gauss_old, 0)[0]    # using new convolution

    t5 = perf_counter()     # time after new gradient calculations

    full_old = canny_compare.edge_detection(img1, 2, 1)     # using old convolution

    t6 = perf_counter()     # time after old edge detector

    full_new = canny_compare.edge_detection(img1, 2, 0)     # using new convolution

    t7 = perf_counter()     # time after new edge detector

    plot_comparison(gauss_old, f"Old Gaussian Smoothing\n{t2-t1:.3f}s",
                    gauss_new, f"New Gaussian Smoothing\n{t3-t2:.3f}s")

    plot_comparison(gradient_old, f"Old Gradient Calculation\n{t4-t3:.3f}s",
                    gradient_new, f"New Gradient Calculation\n{t5-t4:.3f}s")

    plot_comparison(full_old, f"Old Edge Detector\n{t6-t5:.3f}s",
                    full_new, f"New Edge Detector\n{t7-t6:.3f}s")

# Anvit Sinha 10/13/2022
# test the time taken by each function to complete
def time_funcs(img):
    # get time after each step

    t1 = perf_counter()  # initial time

    img1_copy_gradient = canny_compare.grey_scale(img)
    img1_copy_gradient = canny_compare.gaussian_smoothing(img1_copy_gradient, 2)
    img1_copy_gradient, theta = canny_compare.gradient_calc(img1_copy_gradient)

    t2 = perf_counter()  # time after gradient calculations (Sobel)

    plot_comparison(img, "Original Image", img1_copy_gradient, f"Sobel Edge Detection, {t2 - t1:.3f}s")

    img1_copy_nonmax = canny_compare.non_max_suppression(img1_copy_gradient, theta)

    t4 = perf_counter()  # time after non max suppression

    img1_copy_doublet, weak, strong = canny_compare.double_threshold(img1_copy_nonmax)

    t5 = perf_counter()  # time after double threshold

    img1_copy_hysteresis = canny_compare.hysteresis(img1_copy_doublet, weak, strong)

    t6 = perf_counter()  # time after hysteresis

    img1_copy_edge = canny_compare.edge_detection(img, 2)

    t7 = perf_counter()  # time after edge detection (t7 - t6 isolates time for the entire process from start to finish)

    # Plot each figure to compare with the previous step with the time between steps shown
    plot_comparison(img1_copy_gradient, "Sobel Edge Detection", img1_copy_nonmax,
                    f"Non-Max Suppression, {t4 - t2:.3f}s")
    plot_comparison(img1_copy_nonmax, "Non-Max Suppression", img1_copy_doublet,
                    f"Double Thresholding, {t5 - t4:.3f}s")
    plot_comparison(img1_copy_doublet, "Double Thresholding", img1_copy_hysteresis,
                    f"Hysteresis Calculation, {t6 - t5:.3f}s")
    plot_comparison(img, "Original Image", img1_copy_edge, f"Canny Edge Detection, {t7 - t6:.3f}s")


# Method to compare 2 images (plot side by side)
# Copy of testing function from canny_compare.py
def plot_comparison(img1: np.ndarray, name1: str, img2: np.ndarray, name2: str):
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title(name1), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title(name2), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    main()
