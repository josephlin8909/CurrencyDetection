import cv2 as cv
from matplotlib import pyplot as plt


## Anvit Sinha - 10/30/2022
# Function to show part of the testing done on sample images
# to determine the performance of various Template Matching techniques
def main():
    # Read in first sample image
    img1 = cv.imread('old_baht_edge.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image

    # read in the 2nd image
    img3 = cv.imread('100_baht_edge.jpg', 0)
    img4 = img3.copy()

    # read in the template to use
    template = cv.imread('ref.jpg', 0)
    w, h = template.shape[::-1]  # get the width and height of the template

    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for method in methods:

        # template matching on the first image
        img_1_copy = img2.copy()
        method_code = eval(method)

        # Apply template Matching
        res = cv.matchTemplate(img_1_copy, template, method_code)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum since it is needed
        if method_code in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        # plot the result
        cv.rectangle(img_1_copy, top_left, bottom_right, 255, 2)
        plt.subplot(122), plt.imshow(img_1_copy, cmap='gray')
        plt.title('Sample 1'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)

        # do the same thing for the 2nd image to test
        img_3_copy = img4.copy()

        # Apply template Matching
        res = cv.matchTemplate(img_3_copy, template, method_code)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(img_3_copy, top_left, bottom_right, 255, 2)
        plt.subplot(121), plt.imshow(img_3_copy, cmap='gray')
        plt.title('Sample 2'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)

        plt.show()


if __name__ == '__main__':
    main()
