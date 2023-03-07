#%% Import statements

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential 
# from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from PIL import Image
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
# import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

#%%
# make sure that the folder structure of the dataset are of the following: 
#   --Dataset
#       |--Hong Kong Dollar
#       |--Thai Baht
#       |--UAE Dirham
# the data strcuture is a dictionary with key = country currencies
def load_all_dataset(directory):  
    all_dataset = {} 

    currencies = sorted(os.listdir(directory))
    for country in currencies: # loop through country of currencies
        currencies_folder = os.path.join(directory, country)
        for denomination in sorted(os.listdir(currencies_folder)): # loop through denomination values in each currencies
            label = country.split(' ')[0] + " " + denomination + " " + currencies_folder.split(' ')[1]
            denomination_folder = os.path.join(currencies_folder, denomination)
            imdata = [] 
            imlabel = [] 
            for image_file in os.listdir(denomination_folder): # loop through image file in the directory
                imgArr = load_and_resize(os.path.join(denomination_folder, image_file), (256,256))
                all_dataset[label]

def load_and_resize(directory, size):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return np.asarray(image) / 255 

#%% TEST MAIN 
load_all_dataset("/data/plohanak/CurrencyDetection/Dataset")