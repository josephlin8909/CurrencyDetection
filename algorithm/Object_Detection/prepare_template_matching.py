import cv2 as cv
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
from canny import *

def main():
    #create_edge_detection()
    #template_matching()
    collect_images()
    
# Daniel Choi (11/28/22) - Crops a specified rectangle from a provided image to get a template for template matching
def get_template():
    filepathRef = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\ref.jpg"
    filepathBill = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_20"
    img1 = cv.imread(filepathBill, 0)
    img_1_copy = img1.copy()  # make a copy to not modify the original image

    # Tweak these numbers so image shown is the cropped image you want
    y=130
    x=90
    h=180
    w=140
    crop_image = img_1_copy[x:w, y:h]
    cv.imshow("Cropped", crop_image)
    plt.imshow(crop_image, cmap='gray')
    plt.show()
    plt.imsave(filepathRef, crop_image, cmap='gray')

# Runs template matching on all the images in a file and puts the cropped images in another file
def collect_images():
    filepathBill = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_20"
    for filename in os.listdir(filepathBill):
        image_loc = os.path.join(filepathBill, filename)
        template_matching(image_loc, filename)

# Uses template matching to find the location of a template 
def template_matching(filepathBill, filename):
    #filepathBill = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_20\THAI20_63.jpg.jpg"
    filepathRef = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\front_thai_head_ref.jpg"
    filepathEnd = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\noise_for_knn"
    end_loc = os.path.join(filepathEnd, filename)

    img1 = cv.imread(filepathBill, 0)
    img2 = img1.copy()  # make a copy to not modify the original image

    # read in the template to use
    template = cv.imread(filepathRef, 0)
    w, h = template.shape[::-1]  # get the width and height of the template
    # template matching on the first image
    img_1_copy = img2.copy()

    scale_dict = {}

    # Apply template Matching
    for scale in np.linspace(0.5, 1.0, 20): # Try different scales for the image so the algorithm can detect if the image is slightly close up or far away
        temp_img = img_1_copy.copy()
        width = int(temp_img.shape[1] * scale)
        height = int(temp_img.shape[0] * scale)
        dim = (width, height)
        resized = cv.resize(temp_img, dim, interpolation = cv.INTER_NEAREST)
        res = cv.matchTemplate(resized, template, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if scale not in scale_dict:
            scale_dict[scale] = max_val

    print(scale_dict)
    best_scale = max(scale_dict, key=scale_dict.get)
    width = int(img_1_copy.shape[1] * best_scale)
    height = int(img_1_copy.shape[0] * best_scale)
    dim = (width, height)
    resized = cv.resize(img_1_copy, dim, interpolation = cv.INTER_NEAREST)
    res = cv.matchTemplate(resized, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # plot the result
    #cv.rectangle(resized, top_left, bottom_right, 255, 2)
    #plt.subplot(122)
    #plt.imshow(resized, cmap='gray')
    #plt.title('Sample 1')
    #plt.xticks([])
    #plt.yticks([])
    #plt.suptitle("Normalized Cross Correlation")
    #plt.show()

    crop_image = resized[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv.imshow("Cropped", crop_image)
    #plt.imshow(crop_image, cmap='gray')
    plt.show()
    if (max_val >= .345): # Using hand selected value 0.345 to decide which images to save
        plt.imsave(end_loc, crop_image, cmap='gray')

# Run the canny edge detection code on all the images in a file and save it in another file
def create_edge_detection():
    #Change this depending on where the image files are
    filepath20 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\20"
    filepath50 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\50"
    filepath100 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\100"
    filepath500 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\500"
    filepath1000 = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\1000"

    filepath20New = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_20"
    filepath50New = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_50"
    filepath100New = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_100"
    filepath500New = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_500"
    filepath1000New = r"C:\Users\danie\OneDrive - purdue.edu\Currency Detection Team Images\Thai Baht\new_1000"

    allFiles = [filepath100]

    for folder in allFiles:
        for filename in os.listdir(folder): 
            with Image.open(os.path.join(folder, filename), 'r') as image:
                print(filename)
                image = image.resize([256,256])
                a = np.asarray(image)

                # Currently only works with files of size 256x256
                if (np.shape(a)[0] == 256 and np.shape(a)[1] == 256):
                    final = edge_detection(np.array(a), 1)

                    #plt.imshow(final, cmap='gray')
                    file_path = os.path.join(filepath100New, filename + ".jpg") 
                    plt.imsave(file_path, final, cmap='gray')

                    #Display the image if needed
                    #processedImg2 = Image.fromarray(processedImg).convert('RGB')
                    #processedImg2.save("processedImage.jpg")

                else:
                    print("wrong size")

if __name__ == '__main__':
    main()