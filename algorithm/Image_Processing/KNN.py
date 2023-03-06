import matplotlib.pyplot as plt
import math

seal_x_2 = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
seal_y_2 = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
seal_class_2 = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

denom_x_2 = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
denom_y_2 = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
denom_class_2 = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

seal_x_4 = [48.9, 12.2, -57.4, -42.9, -15.5, -89.7, -23.8, -30.6, -42.9, 17.9, -10.3, -50.5, 5.7, -46.3, -78.9, 4.6, 30.6, -89.7, -39.8, -91.2, -7.0, 43.5, 1.5, 60.5, -96.5, -88.8, -97.0, -62.9, 30.3, -27.0, 90.1, -0.1, -83.0, 95.4, 99.3, 31.3, -23.9, 67.4, 40.3, -44.9, 36.0, -96.0, -4.4, -42.8, -37.5, -48.8, -30.7, 13.6, -10.7, -26.1]
seal_y_4 = [72.5, 57.1, 57.0, 42.0, -2.2, -4.3, 48.3, 65.1, -21.3, -68.3, -97.9, 8.0, -36.9, 58.2, -16.7, -6.7, 74.7, -87.5, 56.3, -98.5, 54.2, -18.7, 48.5, 30.8, -20.3, -67.3, -62.6, -76.3, 57.6, 93.7, -30.6, -53.2, -33.1, 88.4, -46.3, 53.0, 75.3, 12.2, 44.3, 71.5, -91.1, -59.3, 78.1, 42.2, -82.4, -32.8, 77.2, -39.1, -15.0, 92.7]
seal_class_4 = [1, 1, 0, 0, 2, 2, 0, 0, 2, 3, 2, 0, 3, 0, 2, 3, 1, 2, 0, 2, 0, 3, 1, 1, 2, 2, 2, 2, 1, 0, 3, 2, 2, 1, 3, 1, 0, 1, 1, 0, 3, 2, 0, 0, 2, 2, 0, 3, 2, 0]

denom_x_4 = [48.9, 12.2, -57.4, -42.9, -15.5, -89.7, -23.8, -30.6, -42.9, 17.9, -10.3, -50.5, 5.7, -46.3, -78.9, 4.6, 30.6, -89.7, -39.8, -91.2, -7.0, 43.5, 1.5, 60.5, -96.5, -88.8, -97.0, -62.9, 30.3, -27.0, 90.1, -0.1, -83.0, 95.4, 99.3, 31.3, -23.9, 67.4, 40.3, -44.9, 36.0, -96.0, -4.4, -42.8, -37.5, -48.8, -30.7, 13.6, -10.7, -26.1]
denom_y_4 = [72.5, 57.1, 57.0, 42.0, -2.2, -4.3, 48.3, 65.1, -21.3, -68.3, -97.9, 8.0, -36.9, 58.2, -16.7, -6.7, 74.7, -87.5, 56.3, -98.5, 54.2, -18.7, 48.5, 30.8, -20.3, -67.3, -62.6, -76.3, 57.6, 93.7, -30.6, -53.2, -33.1, 88.4, -46.3, 53.0, 75.3, 12.2, 44.3, 71.5, -91.1, -59.3, 78.1, 42.2, -82.4, -32.8, 77.2, -39.1, -15.0, 92.7]
denom_class_4 = [1, 1, 0, 0, 2, 2, 0, 0, 2, 3, 2, 0, 3, 0, 2, 3, 1, 2, 0, 2, 0, 3, 1, 1, 2, 2, 2, 2, 1, 0, 3, 2, 2, 1, 3, 1, 0, 1, 1, 0, 3, 2, 0, 0, 2, 2, 0, 3, 2, 0]

# Katherine Sandys (Updated 11/28)
# this function does k k-nearest neighbor
# first four are the numbers of relation (0 to 100), last two are to make sure that
# they are found in the bill image (true or false)
# code returns 0 for thai, 1 for colombia and 2 for neither
def KNN_fucntion_two(colombia_seal, thai_seal, colombia_denom, thai_denom, seal, denom):
    # find the closet point of all the classes, whichever is closer in distance it is said class
    distance_points_denom = []
    distance_points_seal = []
    new_seal_class = -1
    new_denom_class = -1

    if seal:
        # go through all know points to calculate the distance for the seal
        for i in range(len(seal_x_2)):
            dist_x = colombia_seal - seal_x_2[i]
            dist_y = thai_seal - seal_y_2[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_denom.append(total)

        index_smallest_path = distance_points_denom.index(min(distance_points_denom))
        new_seal_class = seal_class_2[index_smallest_path]

        dist = abs(colombia_seal - thai_seal)
        if dist <= 5:
            new_seal_class = 2  # too close to tell

    if denom:
        # go through all know points to calculate the distance for the denomination
        for i in range(len(denom_x_2)):
            dist_x = colombia_denom - denom_x_2[i]
            dist_y = thai_denom - denom_y_2[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_seal.append(total)

        index_smallest_path = distance_points_seal.index(min(distance_points_seal))
        new_denom_class = denom_class_2[index_smallest_path]

        dist = abs(colombia_denom - thai_denom)
        if dist <= 5:
            new_denom_class = 2  # too close to tell

    if seal and denom:  # make sure both are sent in
        if new_denom_class == new_denom_class:
            # classes are the same return it
            return new_denom_class
        else:
            index_smallest_path_seal = min(distance_points_seal)
            index_smallest_path_denom = min(distance_points_denom)
            # the smallest distance class returns in case of a tie
            if index_smallest_path_seal < index_smallest_path_denom:
                return new_seal_class
            elif index_smallest_path_seal > index_smallest_path_denom:
                return new_denom_class
            else:  # if the tie-breaker fails, return neither (2)
                return 4
    elif seal:
        return new_seal_class
    elif denom:
        return new_denom_class


# Katherine Sandys (Updated 2/13)
# this function does k k-nearest neighbor
# first four are the numbers of relation (0 to 100), last two are to make sure that
# they are found in the bill image (true or false)
# code returns 	0 for Thai
            # 1 for Colombia
            # 2 for UAE
            # 3 for Iraq
            # 4 Neither
def KNN_function_four(colombia_seal, thai_seal, x_seal, y_seal, colombia_denom, thai_denom, x_denom, y_denom, seal, denom): #these will be changed later
    #4 classes
    # top right Colombia
    # top left Thai
    # bottom right Iraq -> x
    # bottom left UAE -> y
    # 1.5 on each side of 0 line will be neither

    distance_points_denom = []
    distance_points_seal = []
    new_seal_class = -1
    new_denom_class = -1

    seal_x_test = colombia_seal + x_seal - thai_seal - y_seal
    seal_y_test = colombia_seal - x_seal + thai_seal - y_seal
    denom_x_test = colombia_denom - thai_denom + x_denom - y_denom
    denom_y_test = colombia_denom + thai_denom - x_denom - y_denom

    if seal:
        for i in range(len(seal_x_4)):
            dist_x = seal_x_test - seal_x_4[i]
            dist_y = seal_y_test - seal_y_4[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_seal.append(total)

        index_smallest_path = distance_points_seal.index(min(distance_points_seal))
        new_seal_class = seal_class_4[index_smallest_path]

        dist = abs(seal_x_test - seal_y_test)
        if dist <= 1.5:
            new_seal_class = 4  # too close to tell
        
    if denom:
        for i in range(len(denom_x_4)):
            dist_x = denom_x_test - denom_x_4[i]
            dist_y = denom_y_test - denom_y_4[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_denom.append(total)

        index_smallest_path = distance_points_denom.index(min(distance_points_denom))
        new_denom_class = denom_class_4[index_smallest_path]

        dist = abs(denom_x_test - denom_y_test)
        if dist <= 1.5:
            new_denom_class = 4  # too close to tell


    if seal and denom:  # make sure both are sent in
        if new_denom_class == new_denom_class:
            # classes are the same return it
            return new_denom_class
        else:
            index_smallest_path_seal = min(distance_points_seal)
            index_smallest_path_denom = min(distance_points_denom)
            # the smallest distance class returns in case of a tie
            if index_smallest_path_seal < index_smallest_path_denom:
                return new_seal_class
            elif index_smallest_path_seal > index_smallest_path_denom:
                return new_denom_class
            else:  # if the tie-breaker fails, return neither (2)
                return 4
    elif seal:
        return new_seal_class
    elif denom:
        return new_denom_class