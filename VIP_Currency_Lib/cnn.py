from keras.models import load_model
from skimage.transform import resize
import numpy as np
import cv2

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

# Leng Lohanakakul 12/4/2022
# This function takes a single image input and output a predicted currency
# Function input:
# image: a numpy array with 3 dimensions
# Function output:
# returns a string of the predicted output
def cnn_predict(image):

    if image.shape != (256, 256, 3):
        image = resize_with_padding(image, 256)

    image = np.expand_dims(image, axis=0)
    cnn = load_model('S23_model.h5')
    predictions = cnn.predict(image)

    return(np.argmax(predictions))

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
