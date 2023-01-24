from keras.models import load_model
from skimage.transform import resize
import numpy as np


# Leng Lohanakakul 12/4/2022
# This function takes a single image input and output a predicted currency
# Function input:
# image: a numpy array with 3 dimensions
# Function output:
# returns a string of the predicted output
def cnn_predict(image):

    # Anvit Sinha 12/05/2022
    # Added a check to see if the image is of the correct size
    # and resize if it is not
    if image.shape != (128, 128, 3):
        image = resize(image, (128, 128))

    image_batch = np.expand_dims(image, axis=0)
    cnn = load_model('model3.h5')
    predictions = cnn.predict(image_batch)

    predicted_values = {}

    for label, value in enumerate(predictions[0]):
        if (label not in predicted_values):
            predicted_values[label] = value
        #if value == 1:
        #    return label

    return(max(predicted_values, key=predicted_values.get))

# Anvit Sinha 12/05/2022
# Main function to test the functionality of the CNN model
def __main():

    from PIL import Image
    import matplotlib.pyplot as plt
    image = Image.open('100_baht_test_2.jpg')
    plt.imshow(image)
    plt.show()
    image = image.resize((128, 128))
    plt.imshow(image)
    plt.show()
    image_array = np.asarray(image)
    # image_batch = np.expand_dims(image_array, axis=0)
    print(cnn_predict(image_array))  # print the prediction


if __name__ == '__main__':
    __main()
