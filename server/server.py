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
# Katherine Sandys - Updated the code to work with 4 types
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

    # Create a separate prediction using CNN
    predict_output = cnn.cnn_predict(img)
    print("CNN Classification:")

    # need to be columbian or thai for this function to work right now
    if predict_output in [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16]:
        # Canny Edge Detection image processing techniques are applied.
        processedImg1 = canny.edge_detection(img, 1)

        # Template matching + kNN to classify bill
        thai, col = template_matching.match_template(processedImg1)
        classification = knn.knn_seal(col, thai)
        print("kNN + Template Match Classification:")
        print(classification)


    # Get the bill type and denomination
    # dict = {0: 'Colombian 10000 Pesos', 1: 'Colombian 100000 Pesos', 2: 'Colombian 2000 Pesos', 3: 'Colombian 20000 Pesos', 4: 'Colombian 5000 Pesos', 5: 'Colombian 50000 Pesos', 6: 'Hongkong 10 Dollar', 7: 'Hongkong 100 Dollar', 8: 'Hongkong 1000 Dollar', 9: 'Hongkong 20 Dollar', 10: 'Hongkong 50 Dollar', 11: 'Hongkong 500 Dollar', 12: 'Thai 100 Baht', 13: 'Thai 1000 Baht', 14: 'Thai 20 Baht', 15: 'Thai 50 Baht', 16: 'Thai 500 Baht', 17: 'UAE 10 Dirham', 18: 'UAE 100 Dirham', 19: 'UAE 20 Dirham', 20: 'UAE 200 Dirham', 21: 'UAE 5 Dirham', 22: 'UAE 50 Dirham', 23: 'UAE 500 Dirham'}
    dict = {0: ['Colombian Pesos', 10000], 
            1: ['Colombian Pesos', 100000],
            2: ['Colombian Pesos', 2000],
            3: ['Colombian Pesos', 20000],
            4: ['Colombian Pesos', 5000],
            5: ['Colombian Pesos', 50000],
            6: ['Hongkong Dollar', 10],
            7: ['Hongkong Dollar', 100],
            8: ['Hongkong Dollar', 1000],
            9: ['Hongkong Dollar', 20],
            10: ['Hongkong Dollar', 50],
            11: ['Hongkong Dollar', 500],
            12: ['Thai Baht', 100],
            13: ['Thai Baht', 1000],
            14: ['Thai Baht', 20],
            15: ['Thai Baht', 50],
            16: ['Thai Baht', 500],
            17: ['UAE Dirham', 10],
            18: ['UAE Dirham', 100],
            19: ['UAE Dirham', 20],
            20: ['UAE Dirham', 200],
            21: ['UAE Dirham', 5],
            22: ['UAE Dirham', 50],
            23: ['UAE Dirham', 500]}

    output_vals = dict[predict_output]

    print(output_vals)
    bill_denomination = output_vals[1]
    bill_type_name = output_vals[0]

    bill_type = -1
    if bill_type_name == 'Colombian Pesos':
        bill_type = 0
    elif bill_type_name == 'Hongkong Dollar':
        bill_type = 3
    elif bill_type_name == 'Thai Baht':
        bill_type = 1
    elif bill_type_name == 'UAE Dirham':
        bill_type = 2

    bill_conversion = calc_exchange_rate.calc_exchange_rate(bill_type, bill_denomination)

    print("sending back")
    print("Total Time: --- %s seconds ---" % (time.time() - start))

    # String of important data sent back to app
    output = str(bill_type_name) + "//" + str(bill_denomination) + "//" + str(bill_conversion)

    return output


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
