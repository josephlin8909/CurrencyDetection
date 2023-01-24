import numpy as np

def hysteresis(image: np.array, weak_target: int, strong_target=255):
    # Kavin Sathishkumar 10/1/2022
    # This function identifies the weak pixels in our image which can be edges and discard the remaining.
    # Function inputs:
    # image: Image array to be processed
    # Function output:
    # image_copy: 2D processed image array

    image_row, image_col = image.shape
    image_copy = np.copy(image)  # make a copy of the image to not modify the original image

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            # Anvit Sinha 10/1/2022
            # Refactored for better readability
            print(type(image_copy))
            if image_copy[row, col] != weak_target:
                continue

            if (image_copy[row, col + 1] == strong_target
                    or image_copy[row, col - 1] == strong_target
                    or image_copy[row - 1, col] == strong_target
                    or image_copy[row + 1, col] == strong_target
                    or image_copy[row - 1, col - 1] == strong_target
                    or image_copy[row + 1, col - 1] == strong_target
                    or image_copy[row - 1, col + 1] == strong_target
                    or image_copy[row + 1, col + 1] == strong_target):
                image_copy[row, col] = strong_target
            else:
                image_copy[row, col] = 0

    return image_copy
