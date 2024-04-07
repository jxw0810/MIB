from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler, CSVLogger
from sklearn.preprocessing import LabelBinarizer
from models.models import SimpleVGGNet, VGG16Net, ResNet50Net
from models.VGG import VGG16Net_IBA
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.models import load_model
from gernerate_data import load_cla_data
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
    if epoch % 120 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

INIT_LR = 1e-3
EPOCHS = 200
BS = 16
width = 224
height = 224
depth = 3
target = (width, height)
Name = "BUSI-AN_VGG"
GPU = True
if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = './dataset/Dataset_BUSI_AN/train/images/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(42)
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)

train_x,  train_y, _ = load_cla_data(train_imagePaths, target)
val_x, val_y, _ = load_cla_data(val_imagePaths, target)
test_x, test_y, _ = load_cla_data(test_imagePaths, target)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)

train_y = to_categorical(train_y, 2)
val_y = to_categorical(val_y, 2)
test_y = to_categorical(test_y, 2)

print('train_x shape:', train_x.shape)
# print('val_x shape', val_x.shape)
print('test_x shape', test_x.shape)

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")



model = VGG16Net.build(width=width, height=height, depth=depth, classes=2)
# model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
model.load_weights('BUSI-AN_VGG-100-1.0000.h5')
model.summary()

predictions = model.predict(test_x, batch_size=32)
test = test_y.argmax(axis=1)
print(classification_report(test_y.argmax(axis=1),
      predictions.argmax(axis=1),
      # target_names=lb.classes_,
      digits=3))


print("------Training start------")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/EPOCHS, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', save_best_only='True')

checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', period=20)

H = model.fit_generator(aug.flow(train_x, train_y, batch_size=BS),
                        initial_epoch=0,
                        validation_data=(val_x, val_y), steps_per_epoch=len(train_x) // BS,
                        epochs=EPOCHS,
                        callbacks=[checkpoint_period, checkpoint_period2, reduce_lr, csv_logger])

print("------ testing ------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=lb.classes_,
      digits=3))


print("------saving model------")
model.save(Name+'.h5')
#model.save('./output/cnn.model')

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
plt.savefig('./output/' + Name + '.png')

