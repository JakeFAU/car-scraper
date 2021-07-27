from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import os
import io
import shutil
import cv2

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import callbacks


# create the train and test directories
FILTERED_BASE = "filtered_images"
CLASS_DIR = "classification"
TEST_DIR = os.path.join(CLASS_DIR,"test")
TRAIN_DIR = os.path.join(CLASS_DIR,"train")
IMG_SIZE = (480,480)
car_classes = os.listdir(FILTERED_BASE)
print("Car Class Count=" + str(len(car_classes)))
for car_class in car_classes:
    tr_path = os.path.join(TRAIN_DIR,car_class)
    te_path = os.path.join(TEST_DIR,car_class)
    if os.path.exists(tr_path):
        shutil.rmtree(tr_path)
    if os.path.exists(te_path):
        shutil.rmtree(te_path)
    os.makedirs(tr_path)
    os.makedirs(te_path)

# some counting variables
class_count = 0
train_count = 0
test_count = 0

# train test split
np.random.seed(42)
filtered_classes = os.listdir(FILTERED_BASE)
for filtered_class in filtered_classes:
    class_count = class_count + 1
    filtered_path = os.path.join(FILTERED_BASE,filtered_class)
    filtered_images = os.listdir(filtered_path)
    for filtered_image in filtered_images:
        rnd = np.random.uniform(0,1)
        if rnd <= .2:
            save_path = os.path.join(TEST_DIR, filtered_class, filtered_image)
            test_count = test_count + 1
        else:
            save_path = os.path.join(TRAIN_DIR, filtered_class, filtered_image)
            train_count = train_count + 1
        img_path = os.path.join(filtered_path,filtered_image)
        img = image.load_img(img_path, target_size=IMG_SIZE, interpolation="hamming")
        image.save_img(save_path, img)




# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 9 classes
predictions = Dense(class_count, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# stats
for cc in os.listdir(TRAIN_DIR):
    print("Train " + cc + " has " + str(len(os.listdir(os.path.join(TRAIN_DIR,cc)))))

for cc in os.listdir(TEST_DIR):
    print("Test " + cc + " has " + str(len(os.listdir(os.path.join(TEST_DIR,cc)))))

# some needed variables
img_width, img_height = IMG_SIZE
epochs = 100
batch_size = 4

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Callbacks
cbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

# fix hardcoded numbers
model.fit(
    train_generator,
    steps_per_epoch=train_count // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_count // batch_size,
    callbacks = cbacks)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'],)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit(
    train_generator,
    steps_per_epoch=train_count // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_count // batch_size,
    callbacks = cbacks)

model.save_weights('model_weights.h5')

Y_pred = model.predict(validation_generator, test_count // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Clasnp.argmax(Y_pred, axis=1)sification Report')
target_names = car_classes
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
