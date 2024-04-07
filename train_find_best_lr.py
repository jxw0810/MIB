from sklearn.metrics import classification_report
from keras.utils.np_utils import *
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, label_binarize
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from utils.load_and_save_Model import load_data
from models.models import SimpleVGGNet, VGG16Net
from models.VGG import VGG16Net_IBA
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.models import load_model

import keras.backend as K
import tensorflow as tf
import utils_paths

import numpy as np
import keras
import random
import pickle
import cv2
import os


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch == 0:
        # lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, 1e-6)
        print('The initial lr is {}'.format(1e-6))
    if epoch % 2 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 10)
        print("lr changed to {}".format(lr * 10))
    return K.get_value(model.optimizer.lr)

# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 12
BS = 10
width = 224
height = 224
depth = 3
target = (width, height)

'''
# 是否使用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
'''
# 读取数据和标签
print("------开始读取数据------")

# 拿到图像数据路径，方便后续读取
train_path = '.\\dataset\\ICIS2\\train\\images\\'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(307)
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)

# 遍历训练读取数据
train_x, train_y = load_data(train_imagePaths, target)
val_x, val_y = load_data(val_imagePaths, target)
test_x, test_y = load_data(test_imagePaths, target)


# 转换标签为one-hot encoding格式
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)
train_y = label_binarize(train_y, np.arange(2))
train_y = to_categorical(train_y, 2)
val_y = to_categorical(val_y, 2)
test_y = to_categorical(test_y, 2)
# 数据增强处理
'''
aug = ImageDataGenerator(rotation_range=30,
                         # brightness_range=(0.5, 0.9),
                         rescale=0.8,
                         shear_range=0.2,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode="nearest",
                         )
'''
# 建立卷积神经网络
model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
# model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
# model.summary()
# 损失函数，编译模
#
#
# 型
print("------准备训练网络------")
opt = SGD(lr=1e-6, momentum=0.9, decay=1e-5, nesterov=True)
# opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练网络模型
reduce_lr = LearningRateScheduler(scheduler)

# 自动调节学习率，每100个epcho保存模型

H = model.fit(train_x, train_y, batch_size=BS, initial_epoch=0, workers=4, use_multiprocessing=True,
              validation_data=(val_x, val_y), epochs=EPOCHS,
              callbacks=[reduce_lr])

val_loss = H.history['val_loss']


# 绘制结果曲线
loss = [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]

plt.subplot(121)
new_ticks = np.linspace(0, 7, 8)
plt.xticks(new_ticks)
plt.plot(loss, val_loss)
plt.subplot(122)
plt.scatter(loss, val_loss)
plt.show()
