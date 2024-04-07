from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from IBA.tensorflow_v1 import IBALayer
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from sklearn.metrics import classification_report
from keras.utils.np_utils import *
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer, label_binarize
import pandas as pd
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
# from utils.load_and_save_Model import load_data
from models.models import VGG16Net
from models.VGG import VGG16Net_IBA
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras.backend as K
import utils_paths
import numpy as np
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

def load_cla_data(train_imagePaths, target):

    data = []
    labels = []

    for imagePath in train_imagePaths:

        image = cv2.imread(imagePath)
        image = cv2.resize(image, target)
        data.append(image)
        print(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float")
    #data = data.swapaxes(1, 3)
    data = data / 255.0
    labels = np.array(labels)

    return data, labels

INIT_LR = 1e-3
EPOCHS = 200
BS = 8
width = 224
height = 224
depth = 3
target = (width, height)
Name = "VGG"
GPU = True
if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 建立卷积神经网络
# model = ResNet50Net.build(width=width, height=height, depth=depth, classes=2)
# model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=2)
model = VGG16Net.build(width=width, height=height, depth=depth, classes=2)
# model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
#model = load_model('weights_VGG16_ICIS_100-0.8378.h5')
model.summary()


print("------data loading-----")
train_path = 'dataset/Dataset_BUSI_AN/train/images/'# train_path = './dataset/OCT/train/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(307)
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)


train_x, train_y = load_cla_data(train_imagePaths, target)
val_x, val_y = load_cla_data(val_imagePaths, target)
test_x, test_y = load_cla_data(test_imagePaths, target)

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)
# train_y = label_binarize(train_y, np.arange(2))
train_y = to_categorical(train_y, 2)
val_y = to_categorical(val_y, 2)
test_y = to_categorical(test_y, 2)

# 数据增强处理
aug = ImageDataGenerator(rotation_range=30,
                         # brightness_range=(0.5, 0.9),
                         # rescale=0.8,
                         shear_range=0.2,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         # vertical_flip=True,
                         fill_mode="nearest"
                         )



print("------begin training -----")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=1e-5, nesterov=True)
#opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# opt = Adam(lr=INIT_LR)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period1 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', save_best_only='True', period=20)

checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', period=20)

# earlyStopping = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=1e-8, mode='auto', verbose=1)
#
#
hist = model.fit(train_x, train_y,
                 batch_size=BS,
                 epochs=EPOCHS,
                 validation_data=(val_x, val_y),
                 class_weight="auto",
                 verbose=1,
                 callbacks=[checkpoint_period1, checkpoint_period2, reduce_lr, csv_logger])

# hist = model.fit_generator(aug.flow(train_x, train_y, batch_size=BS),
#                            validation_data=aug.flow(val_x, val_y),
#                            steps_per_epoch=len(train_x) // BS,
#                            class_weight={0: 0.8, 1: 1.2},
#                            validation_steps=5,
#                            workers=6,
#                            epochs=EPOCHS, callbacks=[checkpoint_period1, checkpoint_period2, csv_logger, reduce_lr])


print("------ testing ------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=lb.classes_,
      digits=3))

print("------Saving model------")
# model.save('vgg16_150_ICIS_1-700X700.h5')
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


