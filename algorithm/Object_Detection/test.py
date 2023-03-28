import cv2

# load the image
image = cv2.imread('hk1.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create SIFT object
sift = cv2.SIFT_create()

# detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# draw keypoints on the image
img_keypoints = cv2.drawKeypoints(image, keypoints, None)

# display the image with keypoints
cv2.imshow('SIFT keypoints', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
