import base64
import io
import time
import sys

from PIL import Image
from matplotlib import pyplot as plt

#need to add the path for the folder
# sys.path.insert(0, "/CurrencyDetection_S23/VIP_Currency_Lib")
sys.path.insert(0, "C:/Users/pc/OneDrive - purdue.edu/Documents/GitHub/CurrencyDetection_S23/VIP_Currency_Lib")

from VIP_Currency_Lib import calc_exchange_rate
from VIP_Currency_Lib import cnn
from VIP_Currency_Lib import canny
from VIP_Currency_Lib import template_matching
from VIP_Currency_Lib import knn

#Katherine Sandys (3/6/2023
# testing everything together, using old server code for not
# this will be the server code once everything is put together

image_file = "algorithm/Image_Processing/back_2_crop.png"

#get the image in a Pillow image
receivedImg = Image.open(image_file)
#print(receivedImg.size)

# Resize the image so all input images are same size
old_height = receivedImg.size[0]
old_width = receivedImg.size[1]
new_height = 1280
new_width = (int)((1280 / old_height) * old_width)
size = new_height, new_width
receivedImg = receivedImg.resize(size, Image.NEAREST)
print(receivedImg.size)

receivedImg.save("receivedImage.jpg")
print("saved!")

# Pillow image converted to numpy array
img = plt.imread("receivedImage.jpg", 0)

# Canny Edge Detection image processing techniques are applied.
processedImg1 = canny.edge_detection(img, 1)

# Template matching + kNN to classify bill
thai, col = template_matching.match_template(processedImg1)
classification = knn.knn_seal_2(col, thai) #only doing the seal right now
print("kNN + Template Match Classification:")
print(classification)

# Create a separate prediction using CNN
predict_output = cnn.cnn_predict(img)
print("CNN Classification:")

# Get the bill type and denomination
bill_type = 4  # initially set to neither value

if predict_output in [0, 1, 2, 3, 4]:
    bill_type = 1  # Bill is Thai
elif predict_output in [5, 6, 7, 8, 9]:
    bill_type = 0  # Bill is Columbian

if bill_type == 0:
    bill_type_name = "Columbian Peso"
elif bill_type == 1:
    bill_type_name = "Thai Baht"
elif bill_type == 2:
    bill_type_name = "United Arab Emirates dirham"
elif bill_type == 3:
    bill_type_name = "Hong Kong Dollar"
else:
    bill_type_name = "Unknown"

# denomination initially set to impossible value
bill_denomination = -1

if predict_output == 0:
    bill_denomination = 20
elif predict_output == 1:
    bill_denomination = 50
elif predict_output == 2:
    bill_denomination = 100
elif predict_output == 3:
    bill_denomination = 500
elif predict_output == 4:
    bill_denomination = 1000

print(predict_output)
print(bill_denomination)

bill_conversion = calc_exchange_rate.calc_exchange_rate(
    bill_type, bill_denomination)
