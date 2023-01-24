from algorithm.Object_Detection import template_matching
from VIP_Currency_Lib import canny
import numpy as np
import matplotlib.pyplot as plt


# Anvit Sinha 11/29/2022
# File to test the template matching functions implemented
def main():
    img1 = plt.imread('old_baht_edge.jpg', 0)
    img2 = img1.copy()  # make a copy to not modify the original image

    img2 = canny.edge_detection(img2, 1)  # convert to greyscale

    template = plt.imread('ref.jpg', 0)

    print(np.shape(template), np.shape(img2))  # check the shape of the template and the image

    # Sum of squared differences

    template = plt.imread('baht_ref.jpg', 0)
    template = canny.edge_detection(template, 1)

    res_ssd = template_matching.ssd(img2, template)

    low = np.amin(res_ssd)
    low_loc = np.where(res_ssd == low)

    print(f'Min: {low}\nLocation: {low_loc}')

    # correlation coefficient
    res_ccor = template_matching.corr_coeff(img2, template)  # test corr_coeff

    high = np.amax(res_ccor)
    high_loc = np.where(res_ccor == high)

    print(f'Max: {high}\nLocation: {high_loc}')


if __name__ == '__main__':
    main()
