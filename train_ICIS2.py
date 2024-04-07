from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler, CSVLogger
from sklearn.preprocessing import LabelBinarizer
from models.models import SimpleVGGNet, VGG16Net, ResNet50Net
from models.VGG import VGG16Net_IBA
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.python.client import device_lib
from keras.utils.np_utils import *
import tensorflow as tf
from keras.models import load_model
from keras.optimizers.gradient_descent_v2 import SGD
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

print("------ Data loading  -----")
INIT_LR = 1e-3
EPOCHS = 200
BS = 16
width = 224
height = 224
depth = 3
target = (width, height)
Name = "LE_VGG_IB"
GPU = True
if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = './dataset/LE/train/images/'
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

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
train_y = to_categorical(train_y, 2)
val_y = lb.fit_transform(val_y)
val_y = to_categorical(val_y, 2)
test_y = lb.fit_transform(test_y)
test_y = to_categorical(test_y, 2)

aug = ImageDataGenerator(rotation_range=30,
                         # brightness_range=(0.5, 0.9),
                         # rescale=0.8,
                         shear_range=0.2,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         # vertical_flip=True,
                         fill_mode="nearest",
                         )

# model = SimpleVGGNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))
# model = VGG16Net.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
#model = load_model('weights_VGG16_ICIS_100-0.8378.h5')
model.summary()



print("------ Training start ------ \n")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/EPOCHS, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', save_best_only='True')

checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', period=20)

hist = model.fit_generator(aug.flow(train_x, train_y, batch_size=BS),
                        validation_data=(val_x, val_y),
                        steps_per_epoch=len(train_x) // BS,
                        workers=4,
                        epochs=EPOCHS, callbacks=[checkpoint_period, checkpoint_period2, csv_logger, reduce_lr])


print("------Start predicting------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_, digits=3))

print("------Saving model------")
model_filename = Name + ".h5"
model.save_weights(model_filename)
print('model saved to:', model_filename)

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.plot(N, hist.history["acc"], label="train_acc")
plt.plot(N, hist.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./output/' + Name + '_plot.png')



