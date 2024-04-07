# 导入所需工具�?
from keras.models import load_model
from models.models import VGG16Net
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras.utils.np_utils import *
import numpy as np
from IBA.tensorflow_v1 import IBALayer
import utils_paths
import argparse
import pickle
import os
import cv2

from models.VGG import VGG16Net_IBA

INIT_LR = 1e-3
EPOCHS = 200
BS = 16
width = 224
height = 224
depth = 3
target = (width, height)
Name = "DES_VGGIB"
GPU = True
if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# test_path = 'dataset/DES/train/images/'
test_path = 'dataset/Dataset_BUSI_AN/test/images/'

test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
test_x, test_y = load_data(test_imagePaths, target)

lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
test_y = to_categorical(test_y, 2)

print("------ testing ------")
model = VGG16Net.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
# model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
model.load_weights('BUSI-AN_VGG-100-1.0000.h5')


print("------ testing ------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=lb.classes_,
      digits=3))


# 加载测试数据并进行相同预处理操作
# image = cv2.imread('./cs_image/panda.jpg')
# output = image.copy()
# image = cv2.resize(image, (64, 64))
# scale图像数据
# image = image.astype("float") / 255.0
# 对图像进行拉平操�?
# image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
# 预测
# preds = model.predict(image)

# # 得到预测结果以及其对应的标签
# i = preds.argmax(axis=1)[0]
# label = lb.classes_[i]

# # 在图像中把结果画出来
# text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
# cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
#
# # 绘图
# cv2.imshow("Image", output)
# cv2.waitKey(0)
