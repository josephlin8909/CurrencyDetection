# Joseph Lin 2/5/2023
# Use cv2 to run sobel operator
# added histogram of the sobel edge detection image

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread("iraqi10000ar_180.jpg", cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Calculate the gradient magnitude
result = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = result.astype(np.uint8)

# Save the result
plt.subplot(121), plt.imshow(result, cmap='gray')
plt.title("Soble Edge Detection"), plt.xticks([]), plt.yticks([])
plt.show()

# Plot the histogram
plt.hist(img.ravel(), bins=256, range=(0, 256), fc='k', ec='k')
plt.title("image histogram")
plt.show()
