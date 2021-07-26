from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import requests
import tempfile

URL = "https://www.premierfinancialservices.com/wp-content/uploads/2020/03/1974-Porsche-911-Carrera-Coupe-RMS-.png"
MODEL_WEIGHTS = "model.12-1.33.h5"
base_model = InceptionV3(include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 13 classes
predictions = Dense(13, activation='softmax')(x)
base_model.load_weights(MODEL_WEIGHTS)

SAVE_FILE = tempfile.TemporaryFile()

response = requests.get(URL)
SAVE_FILE.write(response.content)
SAVE_FILE.close()

img = image.load_img(SAVE_FILE, target_size=(224, 224), interpolation="hamming")

preds = base_model.predict(img)
dec_preds = decode_predictions(preds)
for dp in dec_preds:
    print(dp)
