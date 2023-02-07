import matplotlib.pyplot as plt
import math

seal_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
seal_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
seal_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

denom_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
denom_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
denom_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


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
        for i in range(len(seal_x)):
            dist_x = colombia_seal - seal_x[i]
            dist_y = thai_seal - seal_y[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_denom.append(total)

        index_smallest_path = distance_points_denom.index(min(distance_points_denom))
        new_seal_class = seal_class[index_smallest_path]

        dist = abs(colombia_seal - thai_seal)
        if dist <= 5:
            new_seal_class = 2  # too close to tell

    if denom:
        # go through all know points to calculate the distance for the denomination
        for i in range(len(denom_x)):
            dist_x = colombia_denom - denom_x[i]
            dist_y = thai_denom - denom_y[i]
            total = dist_x ** 2 + dist_y ** 2
            total = math.sqrt(total)
            distance_points_seal.append(total)

        index_smallest_path = distance_points_seal.index(min(distance_points_seal))
        new_denom_class = denom_class[index_smallest_path]

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


# Katherine Sandys (Updated 2/6)
# this function does k k-nearest neighbor
# first four are the numbers of relation (0 to 100), last two are to make sure that
# they are found in the bill image (true or false)
# code returns 	0 for Thai
            # 1 for Colombia
            # 2 for UAE
            # 3 for Iraq
            # 4 Neither
def KNN_function_four(): #these will be changed later
    #4 classes
    # top right Colombia
    # top left Thai
    # bottom right Iraq
    # bottom left UAE
    # 1.5 on each side of 0 line will be neither

    #plot new point on graph
    #calculate distance
    #figure out which class to return 
