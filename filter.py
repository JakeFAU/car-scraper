from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os

ID_LIST = ['Model_T', 'ambulance', 'cab','convertible', 'fire_engine', 'garbage_truck','golfcart', 'half_track', 'jeep', 'limousine','minibus', 'minivan', 'moving_van', 'passenger_car','pickup', 'police_van', 'racer', 'recreational_vehicle','school_bus', 'sports_car', 'tow_truck', 'trailer_truck']

def isCar(img_path :str) -> bool:
    model = ResNet50(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224), interpolation="hamming")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]
    predictions = [p[1] for p in decoded_preds]
    return all(item in predictions for item in ID_LIST)


