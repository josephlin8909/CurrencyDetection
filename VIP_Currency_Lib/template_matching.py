import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize


# Anvit Sinha 11/29/2022
# Combined template matching techniques into file
# that is to be added to the library with the entire algorithm
# this version uses opencv for template matching


# Anvit Sinha 12/05/2022
# Performs the template matching using openCV
# and then combines the results from normalized cross correlation and
# correlation coefficient
# Returns the indices for the bottom left and top right of the bounding box
def cv_match_template(input_img: np.array, template: np.array, max_val=False):
    width, height = template.shape[::-1]  # get the width and height of the template

    img_cpy = input_img.copy()  # make a copy of the input image to not modify it

    res_norm_ccor = cv.matchTemplate(img_cpy, template,
                                     cv.TM_CCORR_NORMED)  # get result from normalized cross correlation

    res_ccoeff = cv.matchTemplate(img_cpy, template, cv.TM_CCOEFF_NORMED)  # get result from correlation coefficient

    res_combined = np.add(res_ccoeff, res_norm_ccor) / 2  # combine the results by averaging

    # Anvit Sinha 12/06/2022
    # if the user has requested for the maximum value, send it
    if max_val:
        return res_combined.max()

    # get the x and y coordinates of the max value which represents the top left of the bounding box
    top_left = np.unravel_index(res_combined.argmax(), res_combined.shape)[::-1]

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


# Anvit Sinha 12/05/2022
# Function that performs the template matching based on the
# initial prediction received from the CNN
# expects the image with edge detection done as an input
def match_template(img: np.ndarray, max_value=False):
    src_img = convert(img, 0, 255, np.uint8)  # convert the source image to uint8 for use with template matching

    # to crop to the thai seal
    seal_thai = cv.imread('baht_seal.jpg', 0)
    x_shape = src_img.shape[0] // 6
    y_shape = src_img.shape[1] // 6
    # resize the seal based on the image
    seal_thai = resize(seal_thai, (x_shape, y_shape))
    seal_thai = convert(seal_thai, 0, 255, np.uint8)

    # to crop to the columbian seal
    seal_col = cv.imread('col_seal.jpg', 0)
    seal_col = resize(seal_col, (src_img.shape[0] // 6, src_img.shape[1] // 6))  # resize seal based on the image
    seal_col = convert(seal_col, 0, 255, np.uint8)

    # Anvit Sinha 12/06/2022
    # Added functionality to the function to return the maximum value instead of the
    # cropped image
    if max_value:
        thai_max = cv_match_template(src_img, seal_thai, max_val=max_value)
        col_max = cv_match_template(src_img, seal_thai, max_val=max_value)

        return thai_max, col_max

    else:
        tl_thai, br_thai = cv_match_template(src_img,
                                             seal_thai)  # get the top left and bottom right coords for thai seal
        cropped_thai = crop_img(src_img, tl_thai, br_thai)  # crop the image based on the obtained coordinates
        tl_col, br_col = cv_match_template(src_img,
                                           seal_col)  # get the top left and bottom right coords for thai seal
        cropped_col = crop_img(src_img, tl_col, br_col)  # crop the image based on the obtained coordinates

        return cropped_thai, cropped_col


# Anvit Sinha 11/29/2022
# Main function to test template matching techniques
# private function for testing only
def __main():
    import canny

    # Read in first sample image
    img1 = plt.imread('100_baht_old.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image
    # read in the template to use
    template = plt.imread('baht_seal.jpg', 0)
    template = canny.edge_detection(template, 1)

    src = canny.edge_detection(img2, 2)

    src = convert(src, 0, 255, np.uint8)
    template = convert(template, 0, 255, np.uint8)

    tl, br = cv_match_template(src, template)
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
