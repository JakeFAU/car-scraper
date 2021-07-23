from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import cv2

IMAGE_BASE = "images"
LABELS = []
make_dirs = os.listdir(IMAGE_BASE)
model_dirs = []
for make_dir in make_dirs:
    mdir = os.listdir(os.path.join(IMAGE_BASE,make_dir))
    for d in mdir:
        model_dirs.append(os.path.join(IMAGE_BASE, make_dir, d))

model = ResNet50(weights='imagenet')
for model_dir in model_dirs:
    print("Handling " + model_dir)
    images = os.listdir(model_dir)
    for img_name in images:
        img_path = os.path.join(model_dir, img_name)
        img = image.load_img(img_path, target_size=(224, 224), interpolation="hamming")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]
        for pred in decoded_preds:
            LABELS.append(pred[1])

print(np.unique(LABELS))
