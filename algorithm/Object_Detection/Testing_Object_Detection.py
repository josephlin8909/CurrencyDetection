import cv2
from matplotlib import pyplot as plt

# Kavin Sathishkumar 10/23/2022
# This script is for testing Object Detection using Open-cv.
# image_copy: 2D processed image array after Canny Edge Detection

img = cv2.imread("Thai_currency.png")


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('stop_data.xml')

found = stop_data.detectMultiScale(img_gray,
                                   minSize=(20, 20))


amount_found = len(found)

if amount_found != 0:
    for (x, y, width, height) in found:
        cv2.rectangle(img_rgb, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)


plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()