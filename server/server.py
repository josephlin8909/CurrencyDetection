import base64
import io
import time

from PIL import Image
from flask import Flask
from flask import request
from matplotlib import pyplot as plt

from VIP_Currency_Lib import calc_exchange_rate
from VIP_Currency_Lib import cnn

app = Flask(__name__)


# Base page to test if server is operational
@app.route('/')
def hello_world():
    return 'Hello World!!!'


# Daniel Choi - Function to receive post request from app, process the image, and return it back to the user
@app.route('/post', methods=['POST'])
def image_post():
    # base64 string is received and decoded into byte array
    base64Str = request.form['base64Image']
    print("received!")
    start = time.time()
    decoded_data = base64.b64decode(base64Str)

    print("decoded!")

    # Byte array converted into Pillow image and saved to server
    receivedImg = Image.open(io.BytesIO(decoded_data))
    print(receivedImg.size)

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

    # Our team ran into some issues getting the template matching and kNN working together. Something that can be worked on for future teams.
    # Canny Edge Detection image processing techniques are applied.
    # processedImg1 = canny.edge_detection(img, 1)

    # Template matching + kNN to classify bill
    # thai, col = template_matching.match_template(processedImg1)
    # classification = knn.knn_seal(col, thai)
    # print("kNN + Template Match Classification:")
    # print(classification)

    # Create a separate prediction using CNN
    predict_output = cnn.cnn_predict(img)
    print("CNN Classification:")

    # Get the bill type and denomination

    bill_type = 2  # initially set to neither value

    if predict_output in [0, 1, 2, 3, 4]:
        bill_type = 1 # Bill is Thai
    elif predict_output in [5, 6, 7, 8, 9]:
        bill_type = 0 # Bill is Columbian

    if bill_type == 0:
        bill_type_name = "Columbian Peso"
    elif bill_type == 1:
        bill_type_name = "Thai Baht"
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

    bill_conversion = calc_exchange_rate.calc_exchange_rate(bill_type, bill_denomination)

    print("sending back")
    print("Total Time: --- %s seconds ---" % (time.time() - start))

    # String of important data sent back to app
    output = str(bill_type_name) + "//" + str(bill_denomination) + "//" + str(bill_conversion)

    return output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
