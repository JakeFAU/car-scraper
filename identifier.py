from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
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

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 9 classes
predictions = Dense(13, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(MODEL_WEIGHTS)

SAVE_FILE = tempfile.NamedTemporaryFile()
SAVE_PATH = SAVE_FILE.name

response = requests.get(URL)
SAVE_FILE.write(response.content)

img = image.load_img(SAVE_PATH, target_size=(224, 224), interpolation="hamming")
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

preds = base_model.predict(img)
dec_preds = decode_predictions(preds)[0]
for dp in dec_preds:
    print(dp)

SAVE_FILE.close()
