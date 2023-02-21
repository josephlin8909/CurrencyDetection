# Joseph Lin 02/20/2023
# Split the colored image into bgr channels
# and choose the greatest constract in one of the three channels
# then plot that channel

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('iraqi10000ar_180.jpg')

# Split the image into color channels
b, g, r = cv2.split(img)

# Calculate the histograms of each channel
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# Calculate the contrast of each channel
contrast_b = np.max(hist_b) - np.min(hist_b)
contrast_g = np.max(hist_g) - np.min(hist_g)
contrast_r = np.max(hist_r) - np.min(hist_r)

# Choose the channel with the highest contrast
if contrast_b >= contrast_g and contrast_b >= contrast_r:
    best_channel = b
    cv2.imshow('Blue', b)
elif contrast_g >= contrast_b and contrast_g >= contrast_r:
    best_channel = g
    cv2.imshow('Green', g)
else:
    best_channel = r
    cv2.imshow('Red', r)

cv2.waitKey(0)
# Display the best channel
#plt.hist(best_channel.ravel(), 256, [0, 256])
#plt.show()
