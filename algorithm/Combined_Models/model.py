#%% Import statements

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from PIL import Image
# from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
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

def load_and_resize(directory, size):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return np.asarray(image) / 255 

#%% TEST MAIN 
# load_all_dataset("/data/plohanak/CurrencyDetection/Dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    "/data/plohanak/CurrencyDetection/Dataset",
    validation_split=0.3, 
    subset="training", 
    seed=123, 
    image_size=(256,256),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "/data/plohanak/CurrencyDetection/Dataset",
    validation_split=0.3,
    subset="validation", 
    seed=123, 
    image_size=(256,256), 
    batch_size=32)

class_names = train_ds.class_names 
print(class_names)    

for image_batch, labels_batch in train_ds : 
    print(image_batch.shape) 
    print(labels_batch.shape) 
    break

AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache(). prefetch(buffer_size=AUTOTUNE) 

num_classes = 3

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])

model = Sequential() 
model.add(tf.keras.layers.Rescaling(1./255))
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(256,256,3), name='conv1'))
model.add(Conv2D(64, kernel_size=3, activation='relu', name='conv2'))
model.add(MaxPool2D(pool_size=(4,4), name='pool2'))
model.add(Flatten(name='flat'))
model.add(Dense(3, activation='softmax', name='output'))

# model.summary() 
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
n_epochs = 10
training = model.fit(train_ds, batch_size=32, validation_data=val_ds, epochs=n_epochs)



# %%

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5), squeeze=False) 
# plot the loss value over n_epochs 
ax[0,0].plot(range(1,n_epochs+1), training.history['loss'])
ax[0,0].plot(range(1,n_epochs+1), training.history['val_loss'])
ax[0,0].set_title("losses")
ax[0,0].set_xlabel('epoch') 
ax[0,0].set_ylabel('losses')
ax[0,0].set(ylim=(-0.2,1.2))
# plot the accuracy over n_epochs
ax[0,1].plot(range(1,n_epochs+1), training.history['accuracy'])
ax[0,1].plot(range(1,n_epochs+1), training.history['val_accuracy'])
ax[0,1].set_title("accuracy")
ax[0,1].set_xlabel('epoch') 
ax[0,1].set_ylabel('accuracy')
ax[0,1].set(ylim=(-0.2,1.2))

plt.show() 

print("hello world", type(train_ds))

predictions = model.predict(val_ds)

# score = model.evaluate(x_thai_train, y_thai_train) 
# print(f"Train accuracy of the neural network = {score[1] * 100} %")

# # evaluate the total accuracy of the model 
# score = model.evaluate(x_thai_test, y_thai_test) 
# print(f"Test accuracy of the neural network = {score[1] * 100} %")
# %%
