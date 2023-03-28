import cv2

# Load the image
img = cv2.imread('thai100.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw the keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# Display the image with keypoints
cv2.imshow('SIFT Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
