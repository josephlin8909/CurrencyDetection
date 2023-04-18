#%% Import statements

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import math

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

#%% Leng Lohanakakul -> function that loads the dataset from the directory with separated denomination value
def load_all_dataset(directory, train_size):  
    class_encoding = {} 
    imdata = [] 
    imlabel = [] 

    currencies = sorted(os.listdir(directory))
    offset = 0
    
    for j, country_currency in enumerate(currencies): # loop through country_currency of currencies
        currencies_path = os.path.join(directory, country_currency)
        for i, denomination in enumerate(sorted(os.listdir(currencies_path))): # loop through denomination values in each currencies
            label = country_currency.split(' ')[0] + " " + denomination + " " + country_currency.split(' ')[1] 
            # class_encoding[j] = country_currency 
            class_encoding[i+offset] = label 
            denomination_path = os.path.join(currencies_path, denomination)
            for image_file in os.listdir(denomination_path): # loop through image file in the directory
                imgArr = load_and_resize(os.path.join(denomination_path, image_file), (256,256))
                imdata.append(imgArr) 
                # imlabel.append(j)
                imlabel.append(i+offset)
        offset += len(os.listdir(currencies_path)) # offset of each denomination value
                
    img_train, img_test, lbl_train, lbl_test = train_test_split(imdata, imlabel, train_size=train_size)
    return class_encoding, np.array(img_train), np.array(lbl_train), np.array(img_test), np.array(lbl_test)

# Leng Lohanakakul -> helper function to pre-process the image 
def load_and_resize(directory, size):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_padding(image, size[0])
    return np.asarray(image) / 255 


# Joseph Lin 4/7/2023
# This function takes original image and resize the image to a square image padded with white pixels
def resize_with_padding(img, new_size):
    # get the dimensions of the original image
    height, width, _ = img.shape

    # calculate the new height and width
    if height > width:
        # landscape orientation
        new_width = int(width / height * new_size)
        new_height = new_size
    else:
        # portrait orientation
        new_width = new_size
        new_height = int(new_size / width * height)

    # resize the image while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    # create a white square image of size new_size x new_size
    square_img = 255 * np.ones(shape=[new_size, new_size, 3], dtype=np.uint8)

    # calculate the x and y positions to place the resized image in the center of the square image
    x = (new_size - new_width) // 2
    y = (new_size - new_height) // 2

    # place the resized image in the center of the square image
    square_img[y:y+new_height, x:x+new_width] = resized_img

    return square_img

#%% 
data_dir = "/data/plohanak/CurrencyDetection/Dataset"
encoding, x_train, y_train, x_test, y_test = load_all_dataset(data_dir, 0.9)

split = 809
x_val = x_train[-split:] 
y_val = y_train[-split:] 
x_train = x_train[:-split] 
y_train = y_train[:-split] 

print(f"number of training images = {y_train.shape[0]}")
print(f"number of validation images = {y_val.shape[0]}")
print(f"number of testing images = {y_test.shape[0]}")

print(encoding)

# %%
cnn = Sequential() 
cnn.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(256,256,3), name='conv1'))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', name='conv2'))
cnn.add(MaxPool2D(pool_size=(4,4), name='pool2'))
cnn.add(Flatten(name='flat'))
cnn.add(Dense(len(encoding), activation='softmax', name='output'))

cnn.summary()
# %%

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

n_epochs = 10
training = cnn.fit(x_train, y_train, epochs=n_epochs, validation_data=(x_val, y_val))

cnn.save("/data/plohanak/CurrencyDetection/model.h5")

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), squeeze=False) 
# plot the loss value over n_epochs 
ax[0,0].plot(range(1,n_epochs+1), training.history['loss'], label="Training losses")
ax[0,0].plot(range(1,n_epochs+1), training.history['val_loss'], label="Validation losses")
ax[0,0].set_title("losses")
ax[0,0].set_xlabel('epoch') 
ax[0,0].set_ylabel('losses')
ax[0,0].legend(loc='upper right')
# plot the accuracy over n_epochs
ax[0,1].plot(range(1,n_epochs+1), training.history['accuracy'], label="Training accuracy")
ax[0,1].plot(range(1,n_epochs+1), training.history['val_accuracy'], label="Validation accuracy")
ax[0,1].set_title("accuracy")
ax[0,1].set_xlabel('epoch') 
ax[0,1].set_ylabel('accuracy')
ax[0,1].legend(loc='lower right')

plt.show() 

# %%
import re
thai_regex = re.compile("Thai \d+ Baht")
colombian_regex = re.compile("Colombian \d+ Pesos")
uae_regex = re.compile("UAE \d+ Dirham")
hongkong_regex = re.compile("Hongkong \d+ Dollar")

score = cnn.evaluate(x_test, y_test) 
print(f"Test accuracy of the neural network = {score[1] * 100} %")

# Leng Lohanakakul 11/7/2022
# This code block display an example of correctly and incorrectly classified images 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), squeeze=False) 
predict = cnn.predict(x_test)
# show an example of correcly classified image
for i in range(y_test.shape[0]): 
  j = random.randint(0,199) 
  predicted = np.argmax(predict[j]) 
  # replace the country name in <country>_regex.search to see specific currencies otherwise replace "and" with "or" to see randomized predictions
  if encoding[predicted] == encoding[y_test[j]] or (thai_regex.search(encoding[predicted]) != None): 
    ax[0,0].imshow(np.squeeze(x_test[j]), cmap="gray")
    ax[0,0].set_title(f"classified output = {encoding[predicted]} \n ground truth = {encoding[y_test[j]]}") 
    break

# show an example of misclassified image
for i in range(y_test.shape[0]): 
  j = random.randint(0,199) 
  predicted = np.argmax(predict[j]) 
  if encoding[predicted] != encoding[y_test[j]]: 
    ax[0,1].imshow(np.squeeze(x_test[j]), cmap="gray")
    ax[0,1].set_title(f"classified output = {encoding[predicted]} \n ground truth = {encoding[y_test[j]]}")
    plt.show() 
    break

plt.show() 

# %%
######## delete python directory 

# DIRECT = "/data/plohanak/CurrencyDetection/Dataset/Hongkong Dollar"

# for denomination_folder in sorted(os.listdir(DIRECT)): 
#     files = os.listdir(os.path.join(DIRECT, denomination_folder))
#     files = random.sample(files, 500) # pick the 500 files to keep
#     for images in sorted(os.listdir(os.path.join(DIRECT, denomination_folder))) : 
#         if images not in files: 
#             os.remove(os.path.join(DIRECT, denomination_folder, images))

# %%
image = cv2.imread("/data/plohanak/CurrencyDetection/Dataset/Hongkong Dollar/500/edit4610dollar+26.jpg")
image = resize_with_padding(image, 256)
image = np.expand_dims(image, axis=0)
predict = cnn.predict(image) # output from cnn

encoding = {0: 'Colombian 10000 Pesos', 1: 'Colombian 100000 Pesos', 2: 'Colombian 2000 Pesos', 3: 'Colombian 20000 Pesos', 4: 'Colombian 5000 Pesos', 5: 'Colombian 50000 Pesos', 6: 'Hongkong 10 Dollar', 7: 'Hongkong 100 Dollar', 8: 'Hongkong 1000 Dollar', 9: 'Hongkong 20 Dollar', 10: 'Hongkong 50 Dollar', 11: 'Hongkong 500 Dollar', 12: 'Thai 100 Baht', 13: 'Thai 1000 Baht', 14: 'Thai 20 Baht', 15: 'Thai 50 Baht', 16: 'Thai 500 Baht', 17: 'UAE 10 Dirham', 18: 'UAE 100 Dirham', 19: 'UAE 20 Dirham', 20: 'UAE 200 Dirham', 21: 'UAE 5 Dirham', 22: 'UAE 50 Dirham', 23: 'UAE 500 Dirham'}

print(f"Output from cnn: {predict}")
print(f"Encoded label: {np.argmax(predict)}") 
print(f"Decoded label: {encoding[np.argmax(predict)]}")

# %%
