import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2 as cv


def gradient_calc(img):
    ## Anvit Sinha 9/27/2022
    # This function performs calculations to find the edge intensity and directions.
    # To do this, the function calculates the gradient of the image by performing a convolution
    # of the image with the 3x3 sobel filter

    # Input: The original image as a 2-d matrix
    # Output: The gradient matrix and the slope of the gradient

    # Initialize the 3x3 sobel filter
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # perform the x and y convolution of the image with the filters
    Ix = ndimage.convolve(img, Kx)
    Iy = ndimage.convolve(img, Ky)

    # find the gradient as the hypotenuse of each element in the resultant matrix
    G = np.hypot(Ix, Iy)

    # To ensure that the gradient intensity is between 0 and 255, we divide each element by the max
    # in the matrix and then multiple each value by 255
    G = (G / G.max()) * 255

    # the slope is calculated as the arctan of (Iy/Ix)
    theta = np.arctan2(Iy, Ix)

    return G, theta


def show_plot(img, gradient):
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('3x3 Blurred Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gradient, cmap='gray')
    plt.title('Image with Gradient Intensity'), plt.xticks([]), plt.yticks([])

    plt.savefig('Post_gradient_image.jpg')

    plt.show()


def main():
    img = cv.imread('gaussian_blur_3x3.jpg', 0)

    after_gradient, slope = gradient_calc(img)

    show_plot(img, after_gradient)


if __name__ == '__main__':
    main()
