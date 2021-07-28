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

IMG_SIZE = (480,480)
URL = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fs3.i-micronews.com%2Fuploads%2F2020%2F06%2FToyota1.jpg&f=1&nofb=1"
MODEL_WEIGHTS = "big_model_weights.h5"
CLASS_COUNT = 30

# create the base pre-trained model
base_model = InceptionV3(include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 13 classes
predictions = Dense(CLASS_COUNT, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(MODEL_WEIGHTS)

SAVE_FILE = tempfile.NamedTemporaryFile()
SAVE_PATH = SAVE_FILE.name

response = requests.get(URL)
SAVE_FILE.write(response.content)

img = image.load_img(SAVE_PATH, target_size=IMG_SIZE, interpolation="hamming")
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

results = []
preds = model(img, training=False)
preds = np.squeeze(preds)
catagories =sorted(os.listdir("classification/train"))
for i, cat in enumerate(catagories):
    results.append((cat,preds[i]))
results.sort(key=lambda a: a[1], reverse=True)
print(results)
SAVE_FILE.close()
