import numpy as np
from scipy.signal import convolve, convolve2d
import matplotlib.pyplot as plt


def open_image(path: str):
    # Function inputs:
    # path: Name of file containing the Image to be processed in strings
    # Function output:
    # output: 2D image array
    return plt.imread(path)


def grey_scale(img: np.array):
    # Kavin Sathishkumar 9/25/2022
    # This function converts the input image to greyscale and return the newimage
    # Function inputs:
    # image: RGB image to be processed to grayscale image
    # Function output:
    # output: 2D grayscale image array

    # Create a dictionary to store BT.709 values into Red, Green, and Blue window
    BT709 = {"R": 0.2126, "G": 0.7152, "B": 0.0722}

    # Sum of RGB values according to BT.709 at each pixel
    # Y = 0.2126R + 0.7152G + 0.0722B
    return (img[:, :, 0] * BT709["R"]) + (img[:, :, 1] * BT709["G"]) + (img[:, :, 2] * BT709["B"])


def gaussian_smoothing(img: np.array, sigma: int):
    # Leng Lohanakakul 9/25/2022
    # This function takes an image and sigma value to calculate the gaussian filter
    # and convolve it with the input image
    # Function inputs:
    # image: input grayscale image
    # sigma: sigma value used for Gaussian window
    # Function output:
    # output: returns the convolution between image and the filter

    # gaussian_window = gaussian_matrix(sigma)
    # return convolution(img, gaussian_window)

    # Anvit Sinha 11/12/2022
    # made changes to test using library functions
    gaussian_window = gaussian_matrix(sigma)
    return library_convolve2d(img, gaussian_window)


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


def library_convolve(img: np.array, window: np.array):
    # Anvit Sinha 11/12/2022
    # this function aims to try and replace the brute force convolution function
    # using scipy's convolve method to improve the performance of the algorithm

    # First test: using convolve method

    return convolve(img, window)


def library_convolve2d(img: np.array, window: np.array):
    # Anvit Sinha 11/12/2022
    # this function aims to try and replace the brute force convolution function
    # using scipy's convolve method to improve the performance of the algorithm

    # Second test: using convolve2d method

    return convolve2d(img, window)


def convolution(img: np.array, window: np.array):
    # Leng Lohanakakul 9/25/2022
    # This function performs a 2D convolution between the image and window.
    # Function inputs:
    # image: image to be convolved with a filter
    # window: a window filter

    size_x = img.shape[0]
    size_y = img.shape[1]
    window_size = window.shape[0]
    padding = int((window_size - 1) / 2)
    result_image = np.zeros(img.shape)

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
                    weighted_sum += img[img_row, img_col] * window[x, y]
            result_image[i, j] = weighted_sum

    return result_image


if __name__ == "__main__":
    image = open_image("50_baht.jpg")
    grayscaleImage = grey_scale(image)
    sigmaVal = [1, 2, 3, 4]
    smoothImage = [0] * 4
    for sig in sigmaVal:
        smoothImage[sig - 1] = gaussian_smoothing(grayscaleImage, sig)
    # smoothImage = gaussianSmoothing(grayscaleImage, sigmaVal[3])

    # Testing grayscale image:
    plt.figure()
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(grayscaleImage, cmap='gray')
    plt.title('GrayScaled Image'), plt.xticks([]), plt.yticks([])
    plt.savefig('grayscale.jpg')  # Anvit Sinha 9/27; save image to test gradient calculations
    plt.show()

    # Testing gaussian blur
    plt.figure()
    plt.subplot(221), plt.imshow(grayscaleImage, cmap="gray")
    plt.title('GrayScaled Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(smoothImage[0], cmap='gray')
    plt.title('3X3 window Sigma=1'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(smoothImage[1], cmap='gray')
    plt.title('5X5 window Sigma=2 '), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(smoothImage[2], cmap='gray')
    plt.title('7X7 window Sigma=3'), plt.xticks([]), plt.yticks([])
    plt.savefig('gaussianBlur.jpg')  # Anvit Sinha 9/27; save image to test gradient calculations
    plt.show()

    # Anvit Sinha 9/27/2022
    # Save the image obtained from the 3x3 kernel on its own to test gradient calculations
    plt.figure()
    plt.imshow(smoothImage[0], cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig('gaussian_blur_3x3.jpg')
    plt.show()
