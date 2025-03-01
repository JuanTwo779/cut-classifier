import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np

model = load_model('Image_haircut_classify.keras')
img = 'cut.jpg'

data_cat = ['Burst Fade',
 'Buzz Cut',
 'Caesar Cut',
 'Comb Over',
 'Drop Fade',
 'French Crop',
 'High Fade',
 'Ivy League',
 'Low Fade',
 'Mid Fade',
 'Mid Part',
 'Modern Mullet',
 'Quiff',
 'Side Part',
 'Taper Fade']

img_height = 180
img_width = 180

# image = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
# img_arr = tf.keras.utils.array_to_img(image)
# img_bat = tf.expand_dims(img_arr,0)
# predict = model.predict(img_bat)
# score = tf.nn.softmax(predict)
# print('Haircut in image is a {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

from flask import Flask, request, jsonify
from io import BytesIO

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/test")
def model_test():
    image = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr,0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)

    return 'Haircut in image is a {} with an accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file):
    image = tf.keras.utils.load_img(BytesIO(file.read()), target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr,0)
    return img_bat

@app.post("/predict")
def predict_cut():

    # 1. take in JPG 
    if "file" not in request.files:
        return jsonify({"error":"No file upload"}), 400
    
    file = request.files['file']

    # 2. validate image
    if file.filename == "":  # Handle empty filename
        return jsonify({"error": "No selected file"}), 400
    
    if file and not allowed_file(file.filename):
        return jsonify({"error":"file type not allowed"}), 400

    # 3. fix image
    img_bat = preprocess_image(file)

    # 4. make prediction and produce score
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    
    # 5. return category and score to frontend
    return jsonify({"success": 'Haircut in image is a {} with an accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100) })



