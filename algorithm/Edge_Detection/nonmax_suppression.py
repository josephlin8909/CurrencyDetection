import numpy as np


def non_max_suppression(img, theta):
    # Joseph Lin 9/26/2022
    # Get the dimensions of the image
    # img = a 2D matrix of the image
    # D = a 2D matrix of gradient in radians of every point of the image
    m, n = img.shape
    # Create zero matrix using the same size of the image
    new = np.zeros((m, n), dtype=np.int32)

    # Covert radians to degrees and store in angle
    angle = theta * 180 / np.pi
    # If the angle is less than zero add 180 degrees to make it possitive
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
