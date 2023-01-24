import numpy

from VIP_Currency_Lib import canny
from VIP_Currency_Lib import template_matching
from VIP_Currency_Lib import knn
from VIP_Currency_Lib import cnn

from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np
from PIL import Image


# Anvit Sinha 12/05/2022
# Test the combined functioning of all functions
# and check if the image is classified correctly
def main():
    temp_knn_test()

    cnn_test()


# Anvit Sinha 12/05/2022
# Test the functionality of template matching + knn
def temp_knn_test():
    img = plt.imread('100_baht_test.jpg', 0)  # read image

    src = canny.edge_detection(img, 1)  # perform edge detection

    thai, col = template_matching.match_template(src, max_value=True)
    print(col, thai)
    classification = knn.knn_seal(col * 100, thai * 100)

    print(classification)


# Anvit Sinha 12/05/2022
# Test the functionality of CNN
def cnn_test():
    image = Image.open('100_baht_test.jpg')
    plt.imshow(image)
    plt.show()
    image = image.resize((128, 128))
    plt.imshow(image)
    plt.show()
    image_array = np.asarray(image)
    # image_batch = np.expand_dims(image_array, axis=0)
    print(cnn.cnn_predict(image_array))  # print the prediction


if __name__ == '__main__':
    main()
