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

image = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

print('Haircut in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))