# Joseph Lin 2/3/2023
# Otsu's method

import cv2;
import numpy as np;
import math
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt

# Get the image and plot the image
img = cv2.imread("result.jpg")
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title("Original image"), plt.xticks([]), plt.yticks([])
plt.show()

def grey_scale(image: np.ndarray):
    # Create a dictionary to store BT.709 values into Red, Green, and Blue window
    bt_709 = {"R": 0.2126, "G": 0.7152, "B": 0.0722}

    # Sum of RGB values according to BT.709 at each pixel
    # Y = 0.2126R + 0.7152G + 0.0722B
    return (image[:, :, 0] * bt_709["R"]) + (image[:, :, 1] * bt_709["G"]) + (image[:, :, 2] * bt_709["B"])

# Convert the image to grayscale
img = grey_scale(img)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title("Grayscaled image"), plt.xticks([]), plt.yticks([])

def otsu_method(img):

    # Dictionary that stores all threshold values and their corresponding weighted within-class variance 
    var_sum_dic = {}

    # Initializes the resulting binary image
    img_binary = np.zeros((img.shape[0], img.shape[1]))

    # The total number of pixels of the grayscale image
    pixels_all = img.size

    for thresh in range(256):

        # Calculates the weight of class 1 and class 2
        pixels_class1 = np.count_nonzero(img <= thresh)
        # Class 1 
        weight1 = pixels_class1 / pixels_all
        # Class 2
        weight2 = 1 - weight1

        # Calculates the variance of class 1 and class 2
        # Pixels belong to class 1 and class 2
        pixels1 = img[img <= thresh]
        pixels2 = img[img > thresh]
        # Variance of class 1 and class 2
        var1 = np.var(pixels1)
        var2 = np.var(pixels2)

        # Weighted in-class variance
        var_sum = weight1 * var1 + weight2 * var2

        # Obtains the minimal weighted within-class variance
        if not math.isnan(var_sum):
            var_sum_dic[thresh] = var_sum
            optimal_thresh = min(var_sum_dic, key=var_sum_dic.get)
    
    # The binary image
    img_binary[img > optimal_thresh] = 255

    # Displays the optimal threshold value
    return(img_binary)

# Compute the optimal threshold value using Otsu's method
img = otsu_method(img)

# Plot the final result
plt.subplot(122), plt.imshow(img, cmap='gray')
plt.title("Otsu's method"), plt.xticks([]), plt.yticks([])
plt.show()