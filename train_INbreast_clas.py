from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer
from models.models import SimpleVGGNet, VGG16Net, ResNet34Net, ResNet50Net
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.client import device_lib

from keras.utils.np_utils import *
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import keras.backend as K
import utils_paths

import numpy as np
import argparse
import random
import pickle
import cv2
import os


def scheduler(epoch):
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

def load_data(train_imagePaths, target):

    data = []
    labels = []

    for imagePath in train_imagePaths:

        image = cv2.imread(imagePath)
        image = cv2.resize(image, target)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)


    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels
    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

print("------ loading data -----")
width=700
height=700
target = (width, height)

INIT_LR = 0.01
EPOCHS = 120
BS = 6


train_path = './dataset/INbreast/train/images/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(42)
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)


train_x,  train_y = load_data(train_imagePaths, target)
val_x, val_y = load_data(val_imagePaths, target)
test_x, test_y = load_data(test_imagePaths, target)


# (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
#print("train_x:", train_x)
#print("train_y:", train_y)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
train_y = to_categorical(train_y, 2)
val_y = lb.fit_transform(val_y)
val_y = to_categorical(val_y, 2)
test_y = lb.fit_transform(test_y)
test_y = to_categorical(test_y, 2)

'''
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")



aug = ImageDataGenerator(
                         #brightness_range=(0, 0.8),  
                         #width_shift_range=0.3, 
                         #height_shift_range=0.3,
                         #rescale=0.9,
                         horizontal_flip=True,
                         rotation_range=90,
                         #shear_range=0.2,
                         #zoom_range=0.2,
                         fill_mode="nearest")
'''

#model = SimpleVGGNet.build(width=width, height=height, depth=3, classes=len(lb.classes_))
model = VGG16Net.build(width=width, height=height, depth=3,classes=len(lb.classes_))
#model = load_model('weights_VGG16_INbreast_3_150-0.5122.h5')

print("------ training ------")
opt = SGD(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
#reduce_lr = LearningRateScheduler(scheduler)

checkpoint_period = ModelCheckpoint('weights_VGG16_INbreast_6_{epoch:03d}-{val_acc:.4f}.h5', monitor='val_acc', mode='auto', period=10)

#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, min_lr=1e-7, mode='auto')
'''
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=BS), class_weight={0: 0.66, 1: 1.0}, initial_epoch=0,workers=4, use_multiprocessing=True,
    validation_data=(val_x, val_y), steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS, callbacks=[checkpoint_period, reduce_lr])
'''  
H = model.fit(train_x, train_y, batch_size=BS, initial_epoch=0,workers=8, use_multiprocessing=True,
    validation_data=(val_x, val_y),
    epochs=EPOCHS, callbacks=[checkpoint_period, reduce_lr])
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""


print("------ testing ------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))


print("------saving model------")
model.save('weights_VGG16_150_INbreast_4.h5')
#model.save('./output/cnn.model')
f = open('./output/INbreast2.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./output/VGG16_INbreast_640X832_4.png')

