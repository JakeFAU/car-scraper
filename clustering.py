import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.applications.resnet50 import ResNet50
import cv2
import os, glob, shutil


# Run this cell if you have issues with intitializing cuDNN
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# gather up the images
IMG_DIR = "filtered_images"
imgs = []
img_paths = []
for model in os.listdir(IMG_DIR):
    print("Handling " + model)
    for image in os.listdir(os.path.join(IMG_DIR,model)):
        img_path = os.path.join(IMG_DIR,model,image)
        img = (cv2.resize(cv2.imread(img_path), (224, 224)))
        imgs.append(img)
        img_paths.append(img_path)

images = np.array(np.float32(imgs).reshape(len(imgs), -1)/255)

# use resnet to predict imsage weights
print("Using restnet to gwther image weights")
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
predictions = model.predict(images.reshape(-1, 224, 224, 3))
pred_images = predictions.reshape(images.shape[0], -1)



k = 13
kmodel = KMeans(n_clusters=k, n_jobs=-1, random_state=728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
if os.path.exists("kmeans\output"):
    shutil.rmtree('kmeans\output')
for i in range(k):
	os.makedirs("kmeans\output\cluster" + str(i))
for i in range(len(img_paths)):
	shutil.copy2(img_paths[i], "kmeans\output\cluster"+str(kpredictions[i]))
