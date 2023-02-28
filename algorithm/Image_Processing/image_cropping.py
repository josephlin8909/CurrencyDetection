from PIL import Image
import cv2
from matplotlib import mp
import numpy as np

# Katherine Sandys
# Function inputs:
# image_file: image that is numpy array
# Function output:
# output: array of numpy array of cropped image of a new image size
def image_cropping_mod(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # print("Number of contours detected:", len(contours))

    # print(type(contours))
    # print(contours)

    rectangles = []
    output = []

    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if not (ratio >= 0.9 and ratio <= 1.1):
                #dont want sqaure only rectangles
                rectangles.append([x, y, h, w])

    # print(rectangles)
    #transfer numpy array to image.open thing
    # im = Image.open(r"shapes.png")
    im = Image.fromarray(np.uint8(mp.gist_earth(img)*255))
    width, height = im.size  # get the oringal image size for later

    for i in rectangles:  # list of all the rectangles in the image
        #get coordinated for crop
        left = i[0] - 10
        top = i[1] - 10
        right = i[0] + i[3] + 10
        bottom = i[1] + i[2] + 10
        # print(left, top, right, bottom)
        im1 = im.crop((left, top, right, bottom))
        # im1.show()
        # im1.save('shapes_crop.png')
        output.append(im1)
    
    return output
      
# Katherine Sandys
# Function inputs:
# image_file: image that is numpy array
# Function output:
# output: numpy array of cropped image of the same image size
def image_cropping_same(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # print("Number of contours detected:", len(contours))

    # print(type(contours))
    # print(contours)
    rectangles = []
    output = []

    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if not (ratio >= 0.9 and ratio <= 1.1):
                #dont want sqaure only rectangles
                rectangles.append([x, y, h, w])

    # print(rectangles)
    # im = Image.open(r"shapes.png")
    im = Image.fromarray(np.uint8(mp.gist_earth(img)*255))
    width, height = im.size  # get the oringal image size for later

    for i in rectangles:  # list of all the rectangles in the image
        #get coordinated for crop
        left = i[0] - 10
        top = i[1] - 10
        right = i[0] + i[3] + 10
        bottom = i[1] + i[2] + 10
        # print(left, top, right, bottom)
        im1 = im.crop((left, top, right, bottom))
        # im1.show()
        # im1.save('shapes_crop.png')
        #resize image back
        im.resize((width, height)) #resize image
        output.append(im1)
    
    return output
