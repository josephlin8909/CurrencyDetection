# Anvit Sinha 9/30/2022
# Compiled all functions necessary to perform Canny Edge Detection into one module to help with ease of use
# Also refactored individual functions to improve readability.

import numpy as np
from scipy.signal import convolve2d


def gaussian_smoothing(image: np.ndarray, sigma: int):
    # Leng Lohanakakul 9/25/2022
    # This function takes an image and sigma value to calculate the gaussian filter
    # and convolve it with the input image
    # Function inputs:
    # image: input grayscale image
    # sigma: sigma value used for Gaussian window
    # Function output:
    # output: returns the convolution between image and the filter
    gaussian_window = gaussian_matrix(sigma)

    # Anvit Sinha 11/13/2022
    # changed convolution to use Scipy's convolution function
    result = library_convolve2d(image, gaussian_window)

    return result


def gaussian_matrix(sigma: int):
    # Kavin Sathishkumar 9/25/2022
    # This function applies a gaussian filter on the input image
    # Function inputs:
    # image: image to be passed to the filter
    # sigma: Sigma value (depicts the blur level)
    # Function outputs:
    # window: the filter output containing the window array
    window_size = (2 * sigma) + 1
    x, y = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    gaussian_matrix_distance = np.sqrt((x ** 2) + (y ** 2))
    normal_gaussian = 0.1
    window = np.exp(-(gaussian_matrix_distance ** 2 / (2.0 * sigma ** 2))) * normal_gaussian
    return window


def library_convolve2d(img: np.array, window: np.array):
    # Anvit Sinha 11/12/2022
    # this function replaces the brute force convolution function
    # using scipy's convolve method to improve the performance of the algorithm

    return convolve2d(img, window, mode='same')

def convolution(image: np.ndarray, window: np.ndarray):
    # Leng Lohanakakul 9/25/2022
    # This function performs a 2D convolution between the image and window.
    # Function inputs:
    # image: image to be convolved with a filter
    # window: a window filter

    size_x = image.shape[0]
    size_y = image.shape[1]
    window_size = window.shape[0]
    padding = int((window_size - 1) / 2)
    result_image = np.zeros(image.shape)

    # padded the image to exclude the boundaries to prevent image wrap around
    # iterate through the padded image
    for i in range(padding, size_x - padding):
        for j in range(padding, size_y - padding):
            weighted_sum = 0
            # iterate through each row and column of the window
            for x in range(0, window_size):
                for y in range(0, window_size):
                    # locate and multiply the element in the window with padded image
                    img_row = i + x - padding
                    img_col = j + y - padding
                    weighted_sum += image[img_row, img_col] * window[x, y]
            result_image[i, j] = weighted_sum

    return result_image


def grey_scale(image: np.ndarray):
    # Kavin Sathishkumar 9/25/2022
    # This function converts the input image to greyscale and return the newimage
    # Function inputs:
    # image: RGB image to be processed to grayscale image
    # Function output:
    # output: 2D grayscale image array

    # Create a dictionary to store BT.709 values into Red, Green, and Blue window
    bt_709 = {"R": 0.2126, "G": 0.7152, "B": 0.0722}

    # Sum of RGB values according to BT.709 at each pixel
    # Y = 0.2126R + 0.7152G + 0.0722B
    return (image[:, :, 0] * bt_709["R"]) + (image[:, :, 1] * bt_709["G"]) + (image[:, :, 2] * bt_709["B"])


def gradient_calc(img: np.ndarray):
    # Anvit Sinha 9/27/2022
    # This function performs calculations to find the edge intensity and directions.
    # To do this, the function calculates the gradient of the image by performing a convolution
    # of the image with the 3x3 sobel filter

    # Input: The original image as a 2-d matrix
    # Output: The gradient matrix and the slope of the gradient

    # Initialize the 3x3 sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]],
                       np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]],
                       np.float32)

    # perform the x and y convolution of the image with the filters

    # Anvit Sinha 11/13/2022
    # changed convolution to use Scipy's convolution function
    x_gradient = library_convolve2d(img, sobel_x)
    y_gradient = library_convolve2d(img, sobel_y)

    # find the gradient as the hypotenuse of each element in the resultant matrix
    gradient = np.hypot(x_gradient, y_gradient)

    # To ensure that the gradient intensity is between 0 and 255, we divide each element by the max
    # in the matrix and then multiple each value by 255
    gradient = (gradient / gradient.max()) * 255

    # the slope is calculated as the inverse tangent of (y_gradient/x_gradient)
    theta = np.arctan2(y_gradient, x_gradient)

    return gradient, theta


def non_max_suppression(img: np.ndarray, theta: np.array):
    # Joseph Lin 9/26/2022
    # Get the dimensions of the image
    m, n = img.shape
    # Create zero matrix using the same size of the image
    new = np.zeros((m, n), dtype=np.int32)

    # Find the angle in degrees of the edge of every pixel of the image
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    # Iterate through every pixel of the image
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            a = 255
            b = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                a = img[i, j + 1]
                b = img[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                a = img[i + 1, j - 1]
                b = img[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                a = img[i + 1, j]
                b = img[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                a = img[i - 1, j - 1]
                b = img[i + 1, j + 1]

            if (img[i, j] >= a) and (img[i, j] >= b):
                new[i, j] = img[i, j]
            else:
                new[i, j] = 0

    return new


def double_threshold(img: np.ndarray, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
    # Joseph Lin 9/26/2022
    # Compute high and low threshold value
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    # Get the dimensions of the image
    m, n = img.shape
    # Create zero matrix using the same size of the image
    res = np.zeros((m, n), dtype=np.int32)

    # redefine the weak and strong values of the image that we desire
    weak = np.int32(25)
    strong = np.int32(255)

    # find the coordinates of the pixel that has value larger
    # than the high threshold
    strong_i, strong_j = np.where(img >= high_threshold)

    # find the coordinates of the pixel that has value smaller
    # than the low threshold
    zeros_i, zeros_j = np.where(img < low_threshold)

    # find the coordinates of the pixel that has value larger
    # than the low threshold and smaller than the high threshold
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    # set the pixels that has value larger than the high threshold to strong
    res[strong_i, strong_j] = strong
    # set the pixels that has value smaller than the low threshold to 0
    res[weak_i, weak_j] = weak
    # set the pixels that has value in between low and high threshold to weak
    res[zeros_i, zeros_j] = 0

    return res, weak, strong


def hysteresis(image: np.ndarray, weak_target: int, strong_target=255):
    # Kavin Sathishkumar 10/1/2022
    # This function identifies the weak pixels in our image which can be edges and discard the remaining.
    # Function inputs:
    # image: Image array to be processed
    # Function output:
    # image_copy: 2D processed image array

    image_row, image_col = image.shape
    image_copy = np.copy(image)  # make a copy of the image to not modify the original image

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            # Anvit Sinha 10/1/2022
            # Refactored for better readability
            if image_copy[row, col] != weak_target:
                continue

            if (image_copy[row, col + 1] == strong_target
                    or image_copy[row, col - 1] == strong_target
                    or image_copy[row - 1, col] == strong_target
                    or image_copy[row + 1, col] == strong_target
                    or image_copy[row - 1, col - 1] == strong_target
                    or image_copy[row + 1, col - 1] == strong_target
                    or image_copy[row - 1, col + 1] == strong_target
                    or image_copy[row + 1, col + 1] == strong_target):
                image_copy[row, col] = strong_target
            else:
                image_copy[row, col] = 0

    return image_copy


def edge_detection(img: np.ndarray, sigma: int):
    # Anvit Sinha 10/1/2022
    # This function performs canny edge detection on the given image by using functions defined within this module
    # Input:
    # img: numpy array representation of the image on which edge detection is to be performed
    # sigma: window size to perform gaussian blur on the image as part of the edge detection process
    # Output:
    # img_copy: the resultant image after performing the edge detection steps
    img_copy = np.copy(img)
    img_copy = grey_scale(img_copy)
    img_copy = gaussian_smoothing(img_copy, sigma)
    img_copy, theta = gradient_calc(img_copy)
    img_copy = non_max_suppression(img_copy, theta)
    img_copy, weak, strong = double_threshold(img_copy)
    img_copy = hysteresis(img_copy, weak, strong)

    return img_copy


# Anvit Sinha 10/1/2022
# Main function to test each function against the openCV library
# For testing purposes only; hidden from users who import the package
def __main():
    from matplotlib.pyplot import imread

    img1 = imread('50_baht_inHand.jpg', 0)
    __compare_against_cv(img1, 1)

    img2 = imread('50_baht_test.jpg', 0)
    __compare_against_cv(img2, 1)

    img3 = imread('pesos_test.jpg', 0)
    __compare_against_cv(img3, 2)


# Anvit Sinha 10/1/2022
# Private method to plot an image with a title to test individual functions
# For testing purposes only
def __plot_image(img: np.ndarray, text: str):
    from matplotlib import pyplot as plt

    plt.imshow(img, cmap='gray')
    plt.title(text), plt.xticks([]), plt.yticks([])

    plt.show()


# Anvit Sinha 10/1/2022
# Private method to compare 2 images (plot side by side)
# For testing purposes only
def __plot_comparison(img1: np.ndarray, name1: str, img2: np.ndarray, name2: str):
    from matplotlib import pyplot as plt

    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title(name1), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title(name2), plt.xticks([]), plt.yticks([])

    plt.show()


# Anvit Sinha 10/1/2022
# Private method to compare our functions against openCV at certain landmarks
# For testing purposes only
def __compare_against_cv(img: np.ndarray, sigma: int):
    import cv2 as cv
    from time import perf_counter

    kernel = (2 * sigma) + 1

    # compare gaussian smoothing
    t1 = perf_counter()
    our_grey = grey_scale(img)
    our_smooth = gaussian_smoothing(our_grey, sigma)
    t2 = perf_counter()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    t3 = perf_counter()
    canny_gaussian = cv.GaussianBlur(img_gray, (kernel, kernel), 0)

    __plot_comparison(canny_gaussian, f"Canny Gaussian, {kernel} x {kernel}, {t3 - t2:.4f}s",
                      our_smooth, f"Our Gaussian, {kernel} x {kernel}, {t2 - t1:.4f}s")

    # compare application of Sobel edge detection
    t4 = perf_counter()
    our_gradient, our_theta = gradient_calc(our_grey)
    t5 = perf_counter()
    canny_sobel = np.absolute(cv.Sobel(our_grey, cv.CV_64F, 0, 1, ksize=kernel),  # x - direction Sobel
                              cv.Sobel(our_grey, cv.CV_64F, 1, 0, ksize=kernel))  # y - direction Sobel
    t6 = perf_counter()
    __plot_comparison(canny_sobel, f"Canny Sobel, {kernel} x {kernel}, {t6 - t5:.4f}s",
                      our_gradient, f"Our Sobel, {kernel} x {kernel}, {t5 - t4:.4f}s")

    # Compare Canny Edge detection
    t7 = perf_counter()
    our_canny = edge_detection(img, sigma)
    t8 = perf_counter()
    canny_res = cv.Canny(img, 100, 200)
    t9 = perf_counter()
    __plot_comparison(canny_res, f"Canny Final, {t9 - t8:.4f}s",
                      our_canny, f"Our Final, {kernel} x {kernel}, {t8 - t7:.4f}s")


if __name__ == '__main__':
    __main()
