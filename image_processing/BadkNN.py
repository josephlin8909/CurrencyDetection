from concurrent.futures import process
import numpy as np
#from canny import *
import os
from PIL import Image
import math    
from matplotlib import pyplot as plt
import csv
import pandas as pd

# Daniel Choi and Sohee Kim (10/30/22) - Worked on loading all our test images into a csv file that the object detection 
# algorithms can work with
def createCSV():
    #Change this depending on where the image files are
    filepath20 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\20"
    filepath50 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\50"
    filepath100 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\100"
    filepath500 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\500"
    filepath1000 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\1000"

    allFiles = [filepath20, filepath50, filepath100, filepath500, filepath1000]
    finalAry = []
    col_names = []

    # Turn all images in specified folder into a csv file
    for folder in allFiles:
        for filename in os.listdir(folder): 
            with Image.open(os.path.join(folder, filename), 'r') as image:
                print(filename)
                image = image.resize([100,100])
                a = np.asarray(image)

                # Currently only works with files of size 256x256
                if (np.shape(a)[0] == 100 and np.shape(a)[1] == 100):
                    processedImg = grey_scale(np.array(a))

                    #Display the image if needed
                    #processedImg2 = Image.fromarray(processedImg).convert('RGB')
                    #processedImg2.save("processedImage.jpg")

                    # Each processed image is stored as a nx1 column
                    flatImg = processedImg.reshape(-1,1)
                    flatImgAry = flatImg.tolist()
                    finalAry.append(flatImgAry)
                    filename2 = filename.split('.')[0]
                    col_names.append(filename2)
                else:
                    print("wrong size")

    finalCSVData = np.hstack(finalAry)
    print(np.shape(finalCSVData))
    finalCSV = pd.DataFrame(finalCSVData)
    finalCSV.columns = col_names

    #Change output file location as needed
    #Change output location if using this
    finalCSV.to_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\thai_currency_data.csv', index = None)

    print("Finished!")

# Daniel Choi (11/28/22) - Modified previous function so a dataframe is returned (instead of a dataframe being saved as a csv)
def createDF():
    #Change this depending on where the image files are
    filepathhead = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\cropped_thai_head"
    filepathnoise = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\noise_for_knn"

    allFiles = [filepathhead, filepathnoise]
    finalAry = []
    row_names = []

    # Turn all images in specified folder into a csv file
    for folder in allFiles:
        for filename in os.listdir(folder): 
            with Image.open(os.path.join(folder, filename), 'r') as image:
                    # Each processed image is stored as a nx1 column
                    a = np.asarray(image)
                    flatImg = a.reshape(1,-1)
                    flatImgAry = flatImg.tolist()
                    finalAry.append(flatImgAry)
                    if (folder == filepathhead):
                        filename2 = "head"
                    else:
                        filename2 = "noise"
                    row_names.append(filename2)

    finalCSVData = np.vstack(finalAry)
    print(np.shape(finalCSVData))
    finalDF = pd.DataFrame(finalCSVData, row_names)

    return finalDF

# Daniel Choi (11/8/22) - Got a rough version of kNN working and tested it with mnist handwritten numbers dataset
def EuclideanDistance(inputAry, outputAry):
     total_distance = 0
     for i in range(len(inputAry)):
        total_distance += math.pow((inputAry[i] - outputAry[i]), 2)
     total_distance = math.sqrt(total_distance)
     return total_distance 

def ManhattanDistance(inputAry, outputAry):
    total_distance = 0
    for i in range(len(inputAry)):
        total_distance += math.fabs((inputAry[i] - outputAry[i]))
    return total_distance 

# Assumes the training_data is saved in the following form
# label | pixel at (0,0) | pixel at (1,0) | pixel at (2,0) | ... | pixel at (0,1) | pixel at (1,1) | etc
def kNN(img: np.ndarray, training_data: pd.DataFrame, k: int):
    dist_arr = []

    print(training_data.shape)

    # Compare every pixel in the input image to every pixel in each image in the training dataset
    # Calculate the distance and add it to an array
    #for i in range(training_data.shape[1]):
    for i in range((int) (training_data.shape[0])):
        #print(i)
        #dataset_image = training_data.iloc[:, i].to_numpy()
        dataset_image = training_data.iloc[i,:].to_numpy()
        distance = ManhattanDistance(img, dataset_image) # Changing distance function is one way to modify performance
        dist_arr.append(distance)

    dist_arr2 = np.array(dist_arr)
    print(training_data.shape[0])
    # Get the indices of the k smallest distances 
    smallest_dist_indices = np.argpartition(dist_arr2, k)[:k]

    # At each index where a smallest distance occurs, mark the number of times the class comes up
    predicted_classes = {}
    for index in smallest_dist_indices:
        if (training_data.index[index] not in predicted_classes):
            #predicted_classes[training_data.iloc[index][0]] = 1
            predicted_classes[training_data.index[index]] = 1
        else:
            #predicted_classes[training_data.iloc[index][0]] += 1
            predicted_classes[training_data.index[index]] += 1

        # if (training_data.columns[index].split('_')[0] not in predicted_classes):
        #     predicted_classes[training_data.columns[index].split('_')[0]] = 1
        # else:
        #     predicted_classes[training_data.columns[index].split('_')[0]] += 1

    print(predicted_classes)
    print(max(predicted_classes, key=predicted_classes.get))

    # Return the class that occurs the most number of times
    return max(predicted_classes, key=predicted_classes.get)
    
def split_data():
    # Creates testing and training data
    df = pd.read_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\thai_currency_data.csv')
    training_data = df.sample(frac=0.67, axis=1, ignore_index = False)
    training_data.to_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\kNN_training_data.csv', index = None)
    #print(training_data.columns)
    testing_data = df.drop(labels = training_data.columns, axis = 1)
    testing_data.to_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\kNN_testing_data.csv', index = None)
    print(training_data)
    print(testing_data)

def test_single_image(testing_data, training_data, index):
    #Test a single image
    #testImageFilename = r'C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\500\THAI500_150.jpg'
    #print(training_data.columns[0])

    #index = 175
    img = np.asarray(testing_data.iloc[index])
    #print(img)
    label = testing_data.iloc[index][0]
    prediction = kNN(img, training_data, 5)
    show_prediction(img[1:], prediction, label)

def test_testing_data(testing_data, training_data):
    # Test with all images in testing data
    totalCorrect = 0
    totalTested = 0
    for i in range(testing_data.shape[1]):
        print(i)
        #input_image = training_data.iloc[:, i].to_numpy()
        input_image = np.asarray(testing_data.iloc[i])

        classification = kNN(input_image, training_data, 5)
        print("Actual Label:")
        print(testing_data.iloc[i][0])
        #print(testing_data.columns[i].split('_')[0])
        if (classification == testing_data.iloc[i][0]):
            print("correct prediction")
            totalCorrect += 1
        totalTested += 1
        print("Success Rate so far:")
        print(totalCorrect / totalTested)


    print(totalCorrect / totalTested)

def test_single_image2(testing_data):
    #Test a single image
    testImageFilename = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\THAI20_161.jpg.jpg"
    #print(training_data.columns[0])
    with Image.open(testImageFilename, 'r') as image:
        img = np.asarray(image).reshape(1,-1)
    #label = testing_data.iloc[index][0]
    prediction = kNN(img[0], testing_data, 2)
    print(prediction)
    #show_prediction(img[1:], prediction, label)

def __main():
    #Run these to create a csv file and separate into training and testing data
    dataframe = createDF()
    print(dataframe)
    test_single_image2(dataframe)
    #split_data()
    #training_data = pd.read_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\kNN_training_data.csv')
    #testing_data = pd.read_csv(r'C:\Users\danie\Desktop\CurrencyDetection_F22\algorithm\Object_Detection\kNN_testing_data.csv')

    # Testing with mnist data since it's less complex
    #training_data = pd.read_csv(r'C:\Users\danie\Desktop\Thai Currencies\mnist_train.csv')
    #testing_data = pd.read_csv(r'C:\Users\danie\Desktop\Thai Currencies\mnist_test.csv')

    #test_single_image(testing_data, training_data, 175)

# Personal function to display images
def show_prediction(current_image, prediction, label):
    prediction_str = "kNN Prediction: " + str(prediction)
    label_str = "Ground Truth: " + str(label)
    
    plt.title(prediction_str, loc='left')
    plt.title(label_str, loc='right')
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    __main()
