from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from VIP_Currency_Lib import canny


# Anvit Sinha 11/29/2022
# Combined template matching techniques into file
# that is to be added to the library with the entire algorithm
# this version uses opencv for template matching


# Anvit Sinha 12/05/2022
# Performs the template matching using openCV
# and then combines the results from normalized cross correlation and
# correlation coefficient
# Returns the indices for the bottom left and top right of the bounding box
def match_template(input_img: np.array, template: np.array):
    width, height = template.shape[::-1]  # get the width and height of the template

    img_cpy = input_img.copy()  # make a copy of the input image to not modify it

    res_norm_ccor = cv.matchTemplate(img_cpy, template,
                                     cv.TM_CCORR_NORMED)  # get result from normalized cross correlation

    res_ccoeff = cv.matchTemplate(img_cpy, template, cv.TM_CCOEFF)  # get result from correlation coefficient

    res_combined = np.add(res_ccoeff, res_norm_ccor) // 2  # combine the results by averaging

    # get the x and y coordinates of the max value which represents the top left of the bounding box
    top_left = np.unravel_index(res_combined.argmax(), res_combined.shape)[::-1]

    # top_left = np.unravel_index(res_ccoeff.argmax(), res_ccoeff.shape)[::-1]

    # get the bottom right of the bounding box
    bottom_right = (top_left[0] + width, top_left[1] + height)

    return top_left, bottom_right


# Anvit Sinha 11/29/2022
# Function to crop the image based on where the top left
# and bottom right of the identified square is
def crop_img(img: np.ndarray, top_left, bottom_right):
    # crop the image
    cropped = img[top_left[1]:bottom_right[1],  # get x coordinate
              top_left[0]:bottom_right[0]]  # get y coordinate

    return cropped


# Anvit Sinha 12/05/2022
# Function to convert the data of a numpy ndarray to a target type
# Used to make our canny edge detector results compatible with the
# cv2 template matching code
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# Anvit Sinha 11/29/2022
# Main function to test template matching techniques
# provate function fo testing only
def __main():
    # Read in first sample image
    img1 = plt.imread('100_baht_old.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image
    # read in the template to use
    template = plt.imread('baht_ref.jpg', 0)
    template = canny.edge_detection(template, 1)

    src = canny.edge_detection(img2, 2)

    src = convert(src, 0, 255, np.uint8)
    template = convert(template, 0, 255, np.uint8)

    tl, br = match_template(src, template)
    print(tl, br)

    cropped = crop_img(src, tl, br)

    # plot cropped image
    plt.subplot(122), plt.imshow(cropped, cmap='gray')
    plt.title('After Cropped'), plt.xticks([]), plt.yticks([])

    plt.subplot(121), plt.imshow(src, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == '__main__':
    __main()
