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

URL = "https://cdni.autocarindia.com/ExtraImages/20201124023253_2021-Toyota-Camry-Hybrid-front.jpg"
MODEL_WEIGHTS = "model.12-1.33.h5"

# create the base pre-trained model
base_model = InceptionV3(include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 13 classes
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

results = []
preds = model(img, training=False)
preds = np.squeeze(preds)
catagories =sorted(os.listdir("classification/train"))
for i, cat in enumerate(catagories):
    results.append((cat,preds[i]))

print(results)
SAVE_FILE.close()
