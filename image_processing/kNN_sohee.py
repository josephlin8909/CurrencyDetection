# Load CSV file/NumPy array
# Separate the data to testing and training set (80:20) k = 1 for now,
# but modify k value depending on training/testing set distance results. It is important to note
# that kNN is a lazy algorithm, depending on the location of different classes, the k value may
# be more biased to one class more than another.
# Sort the distances from smallest to largest while using a loop to determine what class
# a data point is in. Obtain the testing & training accuracy, plot the graph using kNN algorithm.

# Sohee Kim (11/7/22) -- My own implementation of the kNN algorithm
# import class from types
# import math
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import sets
from types import new_class
import matplotlib.pyplot as plt
import math

# Sohee Kim (11/8/22) -- Katherine's code of using set perfect points

# Sohee Kim (11/20/22) -- Updated kNN algorithm after deciding to implement based on Katherine's logic, uncommented test image dataset Daniel & I created
sealx = [0, 100, 50, 80, 20, 0, 75]
sealy = [100, 0, 50, 20, 80, 75, 0]
seal_classes = [0, 1, 2, 1, 0, 0, 1]
seal = 1 #hardcoded set values

denominationx = [0, 100, 50, 80, 20, 0, 75]
denominationy = [100, 0, 50, 20, 80, 75, 0]
denomination_classes = [0, 1, 2, 1, 0, 0, 1]
denomination = 0 #hardcoded set values


# Sohee Kim (11/29/22) -- Added few minor additions to my own kNN algorithm after discussing issues
#  on algorithm with Daniel and Katherine
def kNNCurrent(seal, sealx, sealy, seal_classes, denomination, denominationx, denominationy,denomination_classes):
    # find the closest point of all the classes, whichever is closer in distance it is said class
    # go through all know points to calculate the distance for the seal
    if seal:
        dist_seal = euclidean_formula(sealx,sealy)
        # find distance for seal by going through the hard coded points
        seal.append(dist_seal)  # adding all the calculated distance values
        print(seal.append(dist_seal))
        min_index = dist_seal.index(min(dist_seal))  # getting the minimum index of seal
        seal_final = seal_classes[min_index]

    else:
        dist_denomination = euclidean_formula(denominationx,denominationy)
        # find distance for denomination by going through hard coded points
        denomination.append(dist_denomination)  # adding all the calculated distance values
        min_index_two = dist_denomination.index(min(dist_denomination))
        # getting the minimum index of denomination
        denomination_final = denomination_classes[min_index_two]

        # return the greater value
        if denomination > seal:
            return denomination  # 0 means denomination is identified
        if denomination < seal:
            return seal  # 1 means seal is identified
        else:
            return 2  # 2 means that neither were identified


# # Load Dataset
# # Daniel Choi and Sohee Kim (10/30/22) - Worked on loading all our test images into a csv file
# # that the object detection algorithms can work with
# # Initially created a CSV file with Daniel, but will eventually change to just arrays.
#
# # filepath100 = '/home/choidj/Thai Currencies/100'
#
# # Change this depending on where the image files are
# from algorithm.VIP_Currency_Lib.canny import edge_detection
#
# filepath20 = r"C:\Users\danie\Desktop\Thai Currencies/20"
#
# finalAry = []
# col_names = []
#
# # Turn all images in specified folder into a csv file
# for filename in os.listdir(filepath20):
#     with Image.open(os.path.join(filepath20, filename), 'r') as image:
#         print(filename)
#         a = numpy.asarray(image)
#
#         # Currently only works with files of size 256x256
#         if (numpy.shape(a)[0] == 256 and numpy.shape(a)[1] == 256):
#             processedImg = edge_detection(numpy.array(a), 2)
#
#             # Display the image if needed
#             processedImg2 = Image.fromarray(processedImg).convert('RGB')
#             processedImg2.save("processedImage.jpg")
#
#             # Each processed image is stored as a nx1 column
#             flatImg = processedImg.reshape(-1, 1)
#             flatImgAry = flatImg.tolist()
#             finalAry.append(flatImgAry)
#             filename2 = filename.split('.')[0]
#             col_names.append(filename2)
#         else:
#             print("wrong size")
#
# finalCSVData = numpy.hstack(finalAry)
# print(numpy.shape(finalCSVData))
# finalCSV = pandas.DataFrame(finalCSVData)
# finalCSV.columns = col_names
#
# # Change output file location as needed
# finalCSV.to_csv(r'C:\Users\danie\Desktop\Thai Currencies\currency_data.csv', index=None)

# By having several distance equations, we can choose the function that will give most accuracy

# Manhattan distance formula function (x, y)
def manhattan_formula(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += math.fabs((x[i] - y[i]))
    return distance


# Euclidean distance formula function (x, y)
def euclidean_formula(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += math.pow(x[i] - y[i], 2)
    return math.sqrt(distance)


# Cosine distance formula function (x, y)
def cosine_formula(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += (x * y) / (math.fabs(x) * math.fabs(y))
    return distance

# def kNNFirstLogic():
#     # Splitting data, Training to test ratio 80:20
#     #hardcoded kNN function
#
#     x_test, x_train, y_test, y_train = sets(x, y, train_size=0.8, random_state=5)
#     # Setting k value as 1 for test
#     k = KNeighborsClassifier(n_neighbors=1)
#     k.fit(x_train, y_train)
#     k.predict(x_test)
#
# # Plotting the dataset with kNN
# plt.scatter(x, y, c=denomination_classes)
# plt.figure(figsize=(50, 50))
# plt.xlabel('knn neighbors')
# plt.ylabel('accuracy')
# plt.show()
