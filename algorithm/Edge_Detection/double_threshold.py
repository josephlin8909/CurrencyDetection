import numpy as np


def threshold(img, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
    # Joseph Lin 9/26/2022
    # Compute high and low threshold value
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    # Get the dimensions of the image
    M, N = img.shape
    # Create zero matrix using the same size of the image
    res = np.zeros((M, N), dtype=np.int32)

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
