import math


# Anvit Sinha 12/05/2022
# Separated KNN into 2 separate functions
# One for performing KNN on the cropped image of the expected location of the seal
# and one for the expected location of the denomination
# Based on KNN function written by Katherine Sandys, commit: 101eff312f

def knn_seal(colombia_seal, thai_seal):
    # pre-determined values for the seals
    seal_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
    seal_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
    seal_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    distance_points_seal = list()

    # go through all know points to calculate the distance for the seal
    for i in range(len(seal_x)):
        dist_x = colombia_seal - seal_x[i]
        dist_y = thai_seal - seal_y[i]
        total = dist_x ** 2 + dist_y ** 2
        total = math.sqrt(total)
        distance_points_seal.append(total)

    index_smallest_path = distance_points_seal.index(min(distance_points_seal))
    new_seal_class = seal_class[index_smallest_path]

    dist = abs(colombia_seal - thai_seal)
    if dist <= 5:
        new_seal_class = 2  # too close to tell

    return new_seal_class


def knn_denomination(colombia_denom, thai_denom):
    # pre-determined values for the denominations
    denom_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
    denom_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
    denom_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    distance_points_denom = []

    for i in range(len(denom_x)):
        dist_x = colombia_denom - denom_x[i]
        dist_y = thai_denom - denom_y[i]
        total = dist_x ** 2 + dist_y ** 2
        total = math.sqrt(total)
        distance_points_denom.append(total)

    index_smallest_path = distance_points_denom.index(min(distance_points_denom))
    new_denomination_class = denom_class[index_smallest_path]

    dist = abs(colombia_denom - thai_denom)
    if dist <= 5:
        new_denomination_class = 2  # too close to tell

    return new_denomination_class


def __main():
    print(__file__)


if __name__ == '__main__':
    __main()
