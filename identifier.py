from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import requests
import tempfile

URL = "https://www.premierfinancialservices.com/wp-content/uploads/2020/03/1974-Porsche-911-Carrera-Coupe-RMS-.png"
MODEL_WEIGHTS = "model.12-1.33.h5"
base_model = InceptionV3(include_top=False)
base_model.load_weights(MODEL_WEIGHTS)

SAVE_FILE = tempfile.TemporaryFile()

response = requests.get(URL)
SAVE_FILE.write(response.content)
SAVE_FILE.close()

preds = base_model.predict(SAVE_FILE)
dec_preds = decode_predictions(preds)
for dp in dec_preds:
    print(dp)
