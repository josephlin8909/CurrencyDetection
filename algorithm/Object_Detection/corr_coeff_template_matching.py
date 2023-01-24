from matplotlib import pyplot as plt
import numpy as np


# Kavin Sathishkumar - 11/29/2022
# Function to show part corr coff implementation of Template Matching

def main():
    # Read in first sample image
    img1 = plt.imread('old_baht_edge.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image
    # read in the template to use
    template = plt.imread('ref.jpg', 0)

    print(corr_coeff(img2, template))


def corr_coeff(input_image, template, valid_mask=None):
    w, h, x = template.shape[::-1]  # get the width and height of the template
    if valid_mask is None:
        valid_mask = np.ones_like(template)
    window_size = template.shape  # Extract Window size for OutPut Matrix
    R = np.empty((input_image.shape[0] - window_size[0] + 1,  # Initializing R with empty values based on size
                  input_image.shape[1] - window_size[1] + 1))
    for i in range(R.shape[0]):  # Iterating through array shape
        for j in range(R.shape[1]):  # Iterating through array shape
            I1 = (1 / (w * h) * (input_image[i:i + window_size[0], j:j + window_size[1]]))  # Applying equation
            dist = (I1 * ((1 / w * h) * template))  # Applying equation
            R[i, j] = (dist * valid_mask).sum()  # Filling in output matrix
    return R  # Return Output matrix


if __name__ == '__main__':
    main()
