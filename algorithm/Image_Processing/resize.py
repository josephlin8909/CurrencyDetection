import cv2
import numpy as np
# Joseph Lin 4/7/2023
# This function takes original image and resize the image to a square image padded with white pixels

def resize_with_padding(img, new_size):
    # get the dimensions of the original image
    height, width, _ = img.shape

    # calculate the new height and width
    if height > width:
        # landscape orientation
        new_width = int(width / height * new_size)
        new_height = new_size
    else:
        # portrait orientation
        new_width = new_size
        new_height = int(new_size / width * height)

    # resize the image while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    # create a white square image of size new_size x new_size
    square_img = 255 * np.ones(shape=[new_size, new_size, 3], dtype=np.uint8)

    # calculate the x and y positions to place the resized image in the center of the square image
    x = (new_size - new_width) // 2
    y = (new_size - new_height) // 2

    # place the resized image in the center of the square image
    square_img[y:y+new_height, x:x+new_width] = resized_img

    return square_img


img = cv2.imread('hk2.jpg')
img = resize_with_padding(img, 300)
cv2.imshow('resized image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
