#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import os 
from keras.models import Sequential 
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from PIL import Image
from sklearn.model_selection import train_test_split
import random

tf.config.list_physical_devices('GPU')
#%%

#%%
def hk_label_conversion(label) :
  # Joseph Lin 2/18/2023
  # This function converts an integer label of the image to the string equivalent classes 
  # Function input: 
  # label: an integer number specifying the classes of the image 
  # Function output: 
  # returns a string equivalent of the label of the image
  if label == 0 : 
    return "10 dollars"
  elif label == 1: 
    return "20 dollars"
  elif label == 2: 
    return "50 dollars" 
  elif label == 3: 
    return "100 dollars" 
  elif label == 4: 
    return "500 dollars" 
  elif label == 5: 
    return "1000 dollars" 
#%%

#%%
def load_hk_data(directory, getIndividual=None): 
  # Joseph Lin 2/18/2023
  # This function loads the thai dataset from the image folder 
  # do the transformation to the image (resizing and normalizing) 
  # then split the image into training and testing dataset
  # Function inputs: 
  # directory: the path of the folder that contains all the thai images
  # getIndividual: a boolean specifiying which kind of data should be returned

  # image and label for each classes of bank notes
  img_10 = [] 
  lbl_10 = [] 
  img_20 = [] 
  lbl_20 = [] 
  img_50 = [] 
  lbl_50 = [] 
  img_100 = [] 
  lbl_100 = [] 
  img_500 = [] 
  lbl_500 = []
  img_1000 = []
  lbl_1000 = [] 
  # initialize the training size 
  training_size = 0.8

  for image_class in os.listdir(directory) : 
    for image_name in os.listdir(os.path.join(directory, image_class)):
      # open the image at the image file path
      image = cv2.imread(os.path.join(directory, image_class, image_name))
      # resize the image
      image = cv2.resize(image, (300, 225))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # normalize the image
      image = np.array(image) / 255 
      if image_class == "10 dollars": # if the image belongs to 10 hk dollars
        lbel = 0
        img_10.append(image) 
        lbl_10.append(lbel) 
      elif image_class == "20 dollars": # if the image belongs to 20 hk dollars 
        lbel = 1
        img_20.append(image) 
        lbl_20.append(lbel) 
      elif image_class == "50 dollars": # if the image belongs to 50 hk dollars 
        lbel = 2
        img_50.append(image)
        lbl_50.append(lbel)       
      elif image_class == "100 dollars": # if the image belongs to 100 hk dollars 
        lbel = 3
        img_100.append(image) 
        lbl_100.append(lbel) 
      elif image_class == "500 dollars": # if the image belongs to 500 hk dollars 
        lbel = 4
        img_500.append(image) 
        lbl_500.append(lbel) 
      elif image_class == "1000 dollars": # if the image belongs to 1000 hk dollars 
        lbel = 5
        img_1000.append(image) 
        lbl_1000.append(lbel) 
  
  # equally split the dataset into train and test for each classes of images
  hk10_img_train, hk10_img_test, hk10_label_train, hk10_label_test = train_test_split(img_10, lbl_10, train_size=training_size) 
  hk20_img_train, hk20_img_test, hk20_label_train, hk20_label_test = train_test_split(img_20, lbl_20, train_size=training_size) 
  hk50_img_train, hk50_img_test, hk50_label_train, hk50_label_test = train_test_split(img_50, lbl_50, train_size=training_size) 
  hk100_img_train, hk100_img_test, hk100_label_train, hk100_label_test = train_test_split(img_100, lbl_100, train_size=training_size)
  hk500_img_train, hk500_img_test, hk500_label_train, hk500_label_test = train_test_split(img_500, lbl_500, train_size=training_size) 
  hk1000_img_train, hk1000_img_test, hk1000_label_train, hk1000_label_test = train_test_split(img_1000, lbl_1000, train_size=training_size) 
 
  
  # combine the data together into train and test set
  x_train = np.vstack([hk10_img_train, hk20_img_train, hk50_img_train, hk100_img_train, hk500_img_train, hk1000_img_train])
  y_train = np.hstack([hk10_label_train, hk20_label_train, hk50_label_train, hk100_label_train, hk500_label_train, hk1000_label_train])
  x_test = np.vstack([hk10_img_test, hk20_img_test, hk50_img_test, hk100_img_test, hk500_img_test, hk1000_img_test])
  y_test = np.hstack([hk10_label_test, hk20_label_test, hk50_label_test, hk100_label_test, hk500_label_test, hk1000_label_test])
  return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test) 

#%%

#%%
# Leng Lohanakakul 11/7/2022
# Main function for loading the combined and individual dataset

data_dir = '/data/lin1223/Dataset'

# load the dataset and split it into train and test set 
x_thai_train, y_thai_train, x_thai_test, y_thai_test = load_hk_data(data_dir, False)

# number of training and testing dataset 
print(f"number of Thai training images = {y_thai_train.shape[0]}")
print(f"number of Thai testing images = {y_thai_test.shape[0]}")
#%%

#%%
# Leng Lohanakakul 11/7/2022
# Define a convolutional neural network that takes an input shape of 128x128x3
cnn = Sequential() 
cnn.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(225,300, 1), name='conv1'))
cnn.add(Conv2D(64, kernel_size=3, activation='relu', name='conv2'))
cnn.add(MaxPool2D(pool_size=(4,4), name='pool2'))
cnn.add(Flatten(name='flat'))
cnn.add(Dense(6, activation='softmax', name='output'))

cnn.summary()
#%%

#%%
# Leng Lohanakakul 11/7/2022
# Main function for compiling the model and defining the hyperparameters 

# define the optimizer and loss function used for the neural network
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model with training dataset over n_epochs
#x_thai_train = tf.convert_to_tensor(x_thai_train, dtype = tf.int64)
#y_thai_train = tf.convert_to_tensor(y_thai_train, dtype = tf.int64)

n_epochs = 7
training = cnn.fit(x_thai_train, y_thai_train, epochs=n_epochs)

#save model to a file 
# cnn.save("/content/drive/My Drive/VIP/model3_uae.h5")
#%%

#%%
# Leng Lohanakakul 11/7/2022
# This code block displays the performance of the neural network trained over n_epochs
# Measures loss value and accuracy to evalute the performance of cnn 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), squeeze=False) 
# plot the loss value over n_epochs 
ax[0,0].plot(range(1,n_epochs+1), training.history['loss'])
ax[0,0].set_title("losses")
ax[0,0].set_xlabel('epoch') 
ax[0,0].set_ylabel('losses')
# plot the accuracy over n_epochs
ax[0,1].plot(range(1,n_epochs+1), training.history['accuracy'])
ax[0,1].set_title("accuracy")
ax[0,1].set_xlabel('epoch') 
ax[0,1].set_ylabel('accuracy')

plt.show() 
#%%

#%%
# Leng Lohanakakul 11/7/2022
# This code block evaluate the accuracy of the cnn on each testing dataset

# predict the model with testing dataset
predict = cnn.predict(x_thai_test)

# evaluate the total accuracy of the cnn 
score = cnn.evaluate(x_thai_test, y_thai_test) 
print(f"Test accuracy of the neural network = {score[1] * 100} %")

#%%