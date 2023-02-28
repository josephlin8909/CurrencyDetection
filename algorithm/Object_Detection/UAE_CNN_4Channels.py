#%%

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

# %%
def uae_label_conversion(label) :
  # Leng Lohanakakul 2/6/2023
  # This function converts an integer label of the image to the string equivalent classes 
  # Function input: 
  # label: an integer number specifying the classes of the image 
  # Function output: 
  # returns a string equivalent of the label of the image
  if label == 0 : 
    return "5 AED"
  elif label == 1: 
    return "10 AED"
  elif label == 2: 
    return "20 AED" 
  elif label == 3: 
    return "50 AED" 
  elif label == 4: 
    return "100 AED" 
  elif label == 5: 
    return "200 AED" 
  elif label == 6: 
    return "500 AED" 
  
# %%
def load_uae_data(directory, getIndividual=None): 
  # Leng Lohanakakul 11/7/2022
  # This function loads the thai dataset from the image folder 
  # do the transformation to the image (resizing and normalizing) 
  # then split the image into training and testing dataset
  # Function inputs: 
  # directory: the path of the folder that contains all the thai images
  # getIndividual: a boolean specifiying which kind of data should be returned

  # image and label for each classes of bank notes
  img_5 = [] 
  lbl_5 = [] 
  img_10 = [] 
  lbl_10 = [] 
  img_20 = [] 
  lbl_20 = [] 
  img_50 = [] 
  lbl_50 = [] 
  img_100 = [] 
  lbl_100 = [] 
  img_200 = [] 
  lbl_200 = [] 
  img_500 = [] 
  lbl_500 = [] 
  # initialize the training size 
  training_size = 0.8

  for image_class in os.listdir(directory) : 
    for image_name in os.listdir(os.path.join(directory, image_class)):
      # open the image at the image file path
      # image = Image.open(os.path.join(directory, image_class, image_name))
      image = cv2.imread(os.path.join(directory, image_class, image_name))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      image_canny = cv2.GaussianBlur(gray, (3,3), 0)
      image_canny = cv2.Canny(image_canny, 10, 50)
      image_canny = cv2.resize(image_canny, (300,300))
      image = cv2.resize(image, (300,300))
      image = np.dstack((image, image_canny))
      # print(image.shape)
      
      # print("after", image.shape)
      # resize the image to 32x32
      # image.thumbnail((300,300)) 
      
      
      # normalize the image
      image = np.asarray(image) / 255 
      # print("before", image.size)

      if image_class == "AED5": # if the image belongs to 20 thai baht 
        lbel = 0
        img_5.append(image)
        lbl_5.append(lbel) 
      elif image_class == "AED10": # if the image belongs to 50 thai baht 
        lbel = 1
        img_10.append(image) 
        lbl_10.append(lbel) 
      elif image_class == "AED20": # if the image belongs to 100 thai baht 
        lbel = 2
        img_20.append(image) 
        lbl_20.append(lbel) 
      elif image_class == "AED50": # if the image belongs to 500 thai baht 
        lbel = 3
        img_50.append(image) 
        lbl_50.append(lbel)       
      elif image_class == "AED100": # if the image belongs to 1000 thai baht 
        lbel = 4
        img_100.append(image) 
        lbl_100.append(lbel) 
      elif image_class == "AED200": # if the image belongs to 1000 thai baht 
        lbel = 5
        img_200.append(image) 
        lbl_200.append(lbel) 
      elif image_class == "AED500": # if the image belongs to 1000 thai baht 
        lbel = 6
        img_500.append(image) 
        lbl_500.append(lbel) 

  # equally split the dataset into train and test for each classes of images
  uae5_img_train, uae5_img_test, uae5_label_train, uae5_label_test = train_test_split(img_5, lbl_5, train_size=training_size) 
  uae10_img_train, uae10_img_test, uae10_label_train, uae10_label_test = train_test_split(img_10, lbl_10, train_size=training_size) 
  uae20_img_train, uae20_img_test, uae20_label_train, uae20_label_test = train_test_split(img_20, lbl_20, train_size=training_size) 
  uae50_img_train, uae50_img_test, uae50_label_train, uae50_label_test = train_test_split(img_50, lbl_50, train_size=training_size) 
  uae100_img_train, uae100_img_test, uae100_label_train, uae100_label_test = train_test_split(img_100, lbl_100, train_size=training_size)
  uae200_img_train, uae200_img_test, uae200_label_train, uae200_label_test = train_test_split(img_200, lbl_200, train_size=training_size) 
  uae500_img_train, uae500_img_test, uae500_label_train, uae500_label_test = train_test_split(img_500, lbl_500, train_size=training_size) 
   
  # combine the data together into train and test set
  x_train = np.vstack([uae5_img_train, uae10_img_train, uae20_img_train, uae50_img_train, uae100_img_train, uae200_img_train, uae500_img_train])
  y_train = np.hstack([uae5_label_train, uae10_label_train, uae20_label_train, uae50_label_train, uae100_label_train, uae200_label_train, uae500_label_train])
  x_test = np.vstack([uae5_img_test, uae10_img_test, uae20_img_test, uae50_img_test, uae100_img_test, uae200_img_test, uae500_img_test])
  y_test = np.hstack([uae5_label_test, uae10_label_test, uae20_label_test, uae50_label_test, uae100_label_test, uae200_label_test, uae500_label_test])

  return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test) 

# %%
# Leng Lohanakakul 11/7/2022
# Main function for loading the combined and individual dataset

data_dir = '/data/plohanak/CurrencyDetection/Dataset/UAE Currencies'

# load the dataset and split it into train and test set 
x_thai_train, y_thai_train, x_thai_test, y_thai_test = load_uae_data(data_dir, False)

# number of training and testing dataset 
print(f"number of Thai training images = {y_thai_train.shape[0]}")
print(f"number of Thai testing images = {y_thai_test.shape[0]}")
# %%
# Leng Lohanakakul 11/7/2022
# Define a convolutional neural network that takes an input shape of 128x128x3
cnn = Sequential() 
cnn.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(300,300,4), name='conv1'))
cnn.add(Conv2D(64, kernel_size=3, activation='relu', name='conv2'))
cnn.add(MaxPool2D(pool_size=(4,4), name='pool2'))
cnn.add(Flatten(name='flat'))
cnn.add(Dense(7, activation='softmax', name='output'))

cnn.summary()

# %%
# Leng Lohanakakul 11/7/2022
# Main function for compiling the model and defining the hyperparameters 

# define the optimizer and loss function used for the neural network
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model with training dataset over n_epochs 
print(y_thai_train.shape)
print(y_thai_test.shape)

n_epochs = 10
training = cnn.fit(x_thai_train, y_thai_train, epochs=n_epochs)

#save model to a file 
# cnn.save("/content/drive/My Drive/VIP/model3_uae.h5")

# %%
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

# %%
# Leng Lohanakakul 11/7/2022
# This code block evaluate the accuracy of the cnn on each testing dataset

# predict the model with testing dataset
predict = cnn.predict(x_thai_test)

score = cnn.evaluate(x_thai_train, y_thai_train) 
print(f"Train accuracy of the neural network = {score[1] * 100} %")

# evaluate the total accuracy of the cnn 
score = cnn.evaluate(x_thai_test, y_thai_test) 
print(f"Test accuracy of the neural network = {score[1] * 100} %")


# %%
# Leng Lohanakakul 11/7/2022
# This code block display an example of correctly and incorrectly classified images 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), squeeze=False) 

# show an example of correcly classified image
for i in range(y_thai_test.shape[0]): 
  j = random.randint(0,199) 
  predicted = np.argmax(predict[j]) 
  if uae_label_conversion(predicted) == uae_label_conversion(y_thai_test[j]): 
    ax[0,0].imshow(np.squeeze(x_thai_test[j]), cmap="gray")
    ax[0,0].set_title(f"classified output = {uae_label_conversion(predicted)} \n ground truth = {uae_label_conversion(y_thai_test[j])}") 
    break

# show an example of misclassified image
for i in range(y_thai_test.shape[0]): 
  j = random.randint(0,199) 
  predicted = np.argmax(predict[j]) 
  if uae_label_conversion(predicted) != uae_label_conversion(y_thai_test[j]): 
    ax[0,1].imshow(np.squeeze(x_thai_test[j]), cmap="gray")
    ax[0,1].set_title(f"classified output = {uae_label_conversion(predicted)} \n ground truth = {uae_label_conversion(y_thai_test[j])}")
    plt.show() 
    break
# %%
