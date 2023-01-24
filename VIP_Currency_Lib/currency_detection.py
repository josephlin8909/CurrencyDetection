# Anvit Sinha 12/06/2022
# File to combine all functions implemented
# to perform the classification of the currency

# import all files
import canny
import template_matching
import knn
import cnn

from matplotlib import pyplot as plt


# Anvit Sinha 12/06/2022
# Combined classification function
def classify(image):
    src = image.copy()  # make a copy of the image to not modify it

    # edge detection
    src = canny.edge_detection(src, 1)

    # cropping to the templates
    thai, col = template_matching.match_template(src)

    # knn result
    res_k = knn.knn_seal(col, thai)

    # cnn result
    res_c = cnn.cnn_predict(src)

    final = compare_result(res_k, res_c)  # cmpare the results

    return final  # send the final classification


def compare_result(knn_result, cnn_result):
    knn_key = {0: "thai", 1: "col", 2: "neither"}

    cnn_key = {0: "thai", 1: "thai", 2: "thai", 3: "thai", 4: "thai",
               5: "col", 6: "col", 7: "col", 8: "col", 9: "col"}

    # if the results match
    if knn_key[knn_result] == cnn_key[cnn_result]:
        return knn_key[knn_result]

    # if results don't match and KNN classifies as neither, send neither
    elif knn_key[knn_result] == "neither":
        return knn_key[knn_result]

    # if they disagree and both have a classification, CNN takes precedence due to higher accuracy
    else:
        return cnn_key[cnn_result]


def __main():
    print(__file__)


if __name__ == '__main__':
    __main()
