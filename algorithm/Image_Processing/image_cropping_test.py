import os
import random, sys
from PIL import Image  # Importing Image class from PIL module
import cv2
import numpy as np

#load the image in
#image = Image.open("shapes.png")
#run the image processing
#try and find the rectangle
#crop for said area where rectangle is 

#Katherine Sandys
def working():
   img = cv2.imread('back_2.png')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   ret,thresh = cv2.threshold(gray,50,255,0)
   contours,hierarchy = cv2.findContours(thresh, 1, 2)
   # print("Number of contours detected:", len(contours))

   # print(type(contours))
   # print(contours)

   rectangles = []

   for cnt in contours:
      x1,y1 = cnt[0][0]
      approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
      if len(approx) == 4:
         x, y, w, h = cv2.boundingRect(cnt)
         ratio = float(w)/h
         if not (ratio >= 0.85 and ratio <= 1.25):
            #dont want sqaure only rectangles
            rectangles.append([x, y, h, w])

   print(type(rectangles))
   #maybe try getting the average of all rectangles found if more than one is found?
   # rectangle_average = int(sum(rectangles) / len(rectangles))
   rectangle_average = [0, 0, 0, 0]
   for x in rectangles:
      rectangle_average[0] += x[0]
      rectangle_average[1] += x[1]
      rectangle_average[2] += x[2]
      rectangle_average[3] += x[3]

   rectangle_average[0] = int(rectangle_average[0] / len(rectangles)) 
   rectangle_average[1] = int(rectangle_average[1] / len(rectangles))
   print(rectangle_average)

   im = Image.open(r"back_2.png")
   width, height = im.size #get the oringal image size for later

   # for i in rectangles: #list of all the rectangles in the image
   #    #get coordinated for crop 
   #    left = i[0] - 10
   #    top = i[1] - 10
   #    right = i[0] + i[3] + 10
   #    bottom = i[1] + i[2] + 10
   #    # print(left, top, right, bottom)
   #    im1 = im.crop((left, top, right, bottom))
   #    # im1.show()
   #    im1.save('back_2_crop.png')
   #get coordinated for crop 
   left = rectangle_average[0] - 10
   top = rectangle_average[1] - 10
   right = rectangle_average[0] + rectangle_average[3] + 10
   bottom = rectangle_average[1] + rectangle_average[2] + 10
   # print(left, top, right, bottom)
   im1 = im.crop((left, top, right, bottom))
   # im1.show()
   im1.save('back_2_crop.png')

working()

##################
# # Opens a image in RGB mode
def demo2():
   im = Image.open(r"shapes.png")
   
   # Size of the image in pixels (size of original image)
   # (This is not mandatory)
   width, height = im.size
   
   # Setting the points for cropped image
   left = 5
   top = height / 4
   right = 164
   bottom = 3 * height / 4
   
   # Cropped image of above dimension
   # (It will not change original image)
   im1 = im.crop((left, top, right, bottom))
   
   # Shows the image in image viewer
   im1.show()

def test():
   # Load the image
   image = cv2.imread("shapes.png")

   # Convert to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Apply a threshold to the image to make it binary
   _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

   # Find contours in the binary image
   contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Check if there is at least one contour
   if len(contours) > 0:
      # Find the largest contour, which should be the rectangle
      max_contour = max(contours, key=cv2.contourArea)

      # Get the coordinates of the bounding box of the contour
      x, y, w, h = cv2.boundingRect(max_contour)

      # Crop the image to include only the rectangle
      cropped = image[y:y+h, x:x+w]

      # Save the cropped image
      cv2.imwrite("cropped_image.jpg", cropped)

      # Return 1 to indicate success
      print(1)

   # If no contours were found, return 0 to indicate failure
   print(0)

def demo0():
   img = cv2.imread('shapes.png')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   ret, thresh = cv2.threshold(gray, 50, 255, 0)
   contours, hierarchy = cv2.findContours(thresh, 1, 2)
   print("Number of contours detected:", len(contours))

   # print(type(contours))
   for cnt in contours:
      x1, y1 = cnt[0][0]
      approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
      if len(approx) == 4:
         x, y, w, h = cv2.boundingRect(cnt)
         ratio = float(w)/h
         if ratio >= 0.9 and ratio <= 1.1:
            #dont want sqaure
            pass
         else:
            cv2.putText(img, 'Rectangle', (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

   cv2.imshow("Shapes", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
