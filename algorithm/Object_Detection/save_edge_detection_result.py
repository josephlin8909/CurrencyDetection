import numpy as np
from VIP_Currency_Lib import canny
from matplotlib import pyplot as plt


def save_image(source_file, target_file):
    img = plt.imread(source_file)

    larger_side = max(np.shape(img))

    # kernel_sz = int(math.floor(
    #     math.pow(2, larger_side) / (math.comb(larger_side, larger_side // 2) * math.sqrt(2 * math.pi))))
    #
    # print(kernel_sz)
    final = canny.edge_detection(img, 1)

    plt.imshow(final, cmap='gray')
    plt.imsave(target_file, final, cmap='gray')


def main():

    save_image("100_baht_old.jpg", "old_baht_edge.jpg")


if __name__ == '__main__':
    main()
