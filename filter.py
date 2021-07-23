from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import shutil

ID_LIST = ['Model_T', 'ambulance', 'cab','convertible', 'fire_engine', 'garbage_truck','golfcart', 'half_track', 'jeep', 'limousine','minibus', 'minivan', 'moving_van', 'passenger_car','pickup', 'police_van', 'racer', 'recreational_vehicle','school_bus', 'sports_car', 'tow_truck', 'trailer_truck']
MODEL = ResNet50(weights='imagenet')

def isCar(img_path :str) -> bool:
    model = ResNet50(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224), interpolation="hamming")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = MODEL.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]
    predictions = [p[1] for p in decoded_preds]
    is_car = False
    for p in predictions:
        if p in ID_LIST:
            return True
    return False


INPUT_BASE = "images"
OUTPUT_DIR = "filtered_images"
make_dirs = os.listdir(INPUT_BASE)
model_dirs = []
for make_dir in make_dirs:
    mdir = os.listdir(os.path.join(INPUT_BASE,make_dir))
    for d in mdir:
        model_dirs.append(os.path.join(INPUT_BASE, make_dir, d))

for model_dir in model_dirs:
    print("Handling " + model_dir)
    images = os.listdir(model_dir)
    for raw_image in images:
        img_class = raw_image.split("_")[0]
        img_path = os.path.join(model_dir, raw_image)
        if isCar(img_path):
            if not os.path.exists(os.path.join(OUTPUT_DIR,img_class)):
                os.makedirs(os.path.join(OUTPUT_DIR,img_class))
            shutil.copy(img_path,os.path.join(OUTPUT_DIR,img_class,raw_image))

        
