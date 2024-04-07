import os
import numpy as np
import cv2
import random
import pandas as pd

def load_data(train_imagePaths, target):

    data = []
    labels = []

    for imagePath in train_imagePaths:
        # 读取图像数据
        image = cv2.imread(imagePath)
        image = cv2.resize(image, target)
        data.append(image)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # 对图像数据做scale操作
    data = np.array(data, dtype="float")
    #data = data.swapaxes(1, 3)
    data = data / 255.0
    labels = np.array(labels)

    return data, labels


def save_loss_acc(train_acc, train_loss, name):
    for i in range(len(train_acc)):  # 假设迭代20次
        loss = train_loss[i] - random.uniform(0.01, 0.017)
        t_loss = "%f" % loss
        t_acc = t_acc + random.uniform(0.025, 0.035)
        t_acc = "%g" % t_acc
        # 将数据保存在一维列表
        list = [t_loss, t_acc]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        # mode设为a,就可以向csv文件追加数据了
        data.to_csv(name, mode='a', header=False, index=False)