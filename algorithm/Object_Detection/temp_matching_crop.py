import cv2 as cv
from matplotlib import pyplot as plt


## Anvit Sinha - 11/29/2022
# Function to show part of the testing done on sample images
# to show working of template matching to obtain a croppe dimage
def main():
    # Read in sample image
    img1 = cv.imread('old_baht_edge.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image

    img3 = cv.imread('100_baht_edge.jpg', 0)
    img4 = img3.copy()

    # read in the template to use
    template = cv.imread('ref.jpg', 0)
    w, h = template.shape[::-1]  # get the width and height of the template

    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCORR_NORMED']

    for img in [img2, img4]:
        for method in methods:

            # template matching on the first image
            img_copy = img.copy()
            method_code = eval(method)

            # Apply template Matching
            res = cv.matchTemplate(img_copy, template, method_code)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum since it is needed
            if method_code in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + w, top_left[1] + h)

            # plot the result

            # plot cropped image
            cropped = crop_img(img_copy, top_left, bottom_right)
            plt.subplot(122), plt.imshow(cropped, cmap='gray')
            plt.title('After Cropped'), plt.xticks([]), plt.yticks([])

            # draw rectangle on image and crop
            # cv.rectangle(img_copy, top_left, bottom_right, 255, 2)
            plt.subplot(121), plt.imshow(img_copy, cmap='gray')
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.suptitle(method)

            plt.show()


# Anvit Sinha 11/29/2022
# Function to crop the image based on where the top left
# and bottom right of the identified square is
def crop_img(img, top_left, bottom_right):
    # crop the image
    cropped = img[top_left[1]:bottom_right[1],  # get x coordinate
                  top_left[0]:bottom_right[0]]  # get y coordinate

    return cropped


if __name__ == '__main__':
    main()
