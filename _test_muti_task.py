import keras.backend

from gernerate_data import load_clas_seg_data
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_curve
from keras.utils.np_utils import *
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, auc, roc_curve
from models.MLT_net import MTL_classic
from sklearn.preprocessing import LabelBinarizer, label_binarize
from utils.losses import dice_coef_loss, dice_coef, dice_coef_loss, focal_tversky, p_r_f1_iou
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import keras.backend as K
import utils_paths
import numpy as np
import cv2 as cv
import pickle
import os


INIT_LR = 5e-3
EPOCHS = 200
batch_size = 4
depth = 3
img_size = 224
Name = "Multi_IB_BUSI_0.4"
GPU = True
target = (img_size, img_size)

if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print("------------------------------------------------ Reading data ------------------------------------------------")
#
testX_dir = 'dataset/Dataset_BUSI_AN/test/images/'
weight_path = 'wights/Multi_IB_BUSI_0.901.h5'

model = MTL_classic(img_size, img_size, depth, nClasses=2)
model.summary()


model.load_weights('MT-IB_BUSI.h5')


test_x, test_c_y, test_s_y = load_clas_seg_data(testX_dir, target)


lb = LabelBinarizer()
test_c_y = lb.fit_transform(test_c_y)
test_c_y = to_categorical(test_c_y, 2)


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='classification_output_loss', factor=0.1, patience=10, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period1 = ModelCheckpoint(Name + '-{epoch:03d}-{val_segmentation_output_acc:.4f}.h5',
                                     monitor='classification_output_acc', mode='auto', save_best_only='True')

checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_segmentation_output_acc:.4f}.h5',
                                     monitor='segmentation_output_loss', mode='auto', period=20)


# define loss and compile the model

print("------------------------------------------------ begin training ------------------------------------------------")
opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.01)

model.compile(loss={'segmentation_output': focal_tversky, "classification_output": binary_crossentropy},
              loss_weights={'segmentation_output': 0.3, "classification_output": 0.7},
              optimizer=opt,
              metrics={'segmentation_output': ['accuracy', dice_coef], "classification_output": ['accuracy']})


print("------ testing ------")
predictions_c, predictions_s = model.predict(test_x, batch_size=32)
test_c = test_c_y.argmax(axis=1)
print(classification_report(test_c_y.argmax(axis=1),
      predictions_c.argmax(axis=1),
      digits=4))

print("------------------------------------------------ Segmentation testing ------------------------------------------------")

# evaluate the model
# loss = model.evaluate(test_x, [test_c_y, test_s_y], verbose=0)
loss, cla_loss, seg_loss, cla_acc, seg_acc, seg_dice_coef = model.evaluate(test_x, [test_c_y, test_s_y], verbose=0)
print('Test total loss:', loss)
print('Test classification loss:', cla_loss)
print('Test classification accuracy:', cla_acc)

print('Test segmentation loss:', seg_loss)
print('Test segmentation accuracy:', seg_acc)
print('Test segmentation dice_coef:', seg_dice_coef)

preds_c, preds_s = model.predict(test_x, batch_size=8, verbose=1)
test_mask = test_s_y.flatten()
pred_mask = preds_s.flatten()
fpr, tpr, thresholds = roc_curve(test_mask, pred_mask, pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)


import matplotlib.pyplot as plt
lw = 2
plt.figure(figsize=(10, 10))
# ls：折线图的线条风格; lw：折线图的线条宽度; label：标记图内容的标签文本
# 假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
# 设置图标
# plt.legend(loc="lower right")
plt.legend(loc="upper right")
plt.show()
plt.savefig('VGG_BUSI-ROC.png', dpi=100, transparent=False)

# Precision, Recall, accuracy, F1, IoU = p_r_f1_iou(test_s_y, preds_s)
# print('Precision:', Precision)
# print('Recall:', Recall)
# print('m_Iou:', IoU)
# print('F1_score:', F1)
