from keras.models import load_model


# Leng Lohanakakul 12/4/2022
# This function takes a single image input and output a predicted currency
# Function input:
# image: a numpy array with 3 dimensions
# Function output:
# returns a string of the predicted output
def cnn_predict(image):
    import numpy as np
    image_batch = np.expand_dims(image, axis=0)
    cnn = load_model('/model3.h5')
    predictions = cnn.predict(image_batch)
    for label, value in enumerate(predictions[0]):
        if value == 1:
            return label
