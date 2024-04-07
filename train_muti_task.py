from gernerate_data import load_clas_seg_data
import tensorflow as tf
from sklearn.metrics import classification_report, auc, roc_curve
from keras.utils.np_utils import *
from keras.callbacks import LearningRateScheduler
from models.MLT_net import MIB_Net, Multi_IB, Multi_task_VGG16_5, Multi_task_VGG16_4, \
    Multi_task_Unet_finetune, Multi_task_Unet_pool3_finetune, Multi_task_Unet_pool4_finetune
from sklearn.preprocessing import LabelBinarizer, label_binarize
from utils.losses import dice_coef_loss, dice_coef, dice_p_bce, dice_p_focal, tversky_loss, focal_loss, focal_tversky, \
    p_r_f1_iou, generalized_dice_coeff, generalized_dice_loss
from keras.losses import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizer_v1 import Adam
import matplotlib.pyplot as plt
import keras.backend as K
import utils_paths
import numpy as np
import pickle
import os


INIT_LR = 5e-4
EPOCHS = 200
batch_size = 8
depth = 3
img_size = 224
Name = "MIB_LE_0.3"
GPU = True
target = (img_size, img_size)

if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print("------------------------------------------------ Reading data ------------------------------------------------")
trainX_dir = 'dataset/LE/test/images/'
#trainX_dir = 'dataset/Dataset_BUSI/train/images/'
valX_dir = trainX_dir.replace('train', 'val')
testX_dir = trainX_dir.replace('train', 'test')
weight_path = 'wights/UNet_BUSI.h5'

# model = Multi_task_Unet_pool3_finetune(input_width=img_size, input_height=img_size, depth=depth, nClasses=2, weight_path=weight_path)
# model = Multi_task_Unet_finetune(input_width=img_size, input_height=img_size, depth=depth, nClasses=2)
# model = Multi_task_VGG16_5(input_width=img_size, input_height=img_size, depth=depth, nClasses=2)
# model = Multi_task(input_width=img_size, input_height=img_size, depth=depth, nClasses=2)
# model = Multi_IB(input_width=img_size, input_height=img_size, depth=depth, nClasses=2)
model = MIB_Net(input_width=img_size, input_height=img_size, depth=depth, nClasses=2)
model.summary()
# model.load_weights('Multi_IB_BUSI_0.4-040-0.9199.h5')


train_x,  train_c_y, train_s_y = load_clas_seg_data(trainX_dir, target)
val_x, val_c_y, val_s_y = load_clas_seg_data(valX_dir, target)
test_x, test_c_y, test_s_y = load_clas_seg_data(testX_dir, target)


lb = LabelBinarizer()
train_c_y = lb.fit_transform(train_c_y)
val_c_y = lb.fit_transform(val_c_y)
test_c_y = lb.fit_transform(test_c_y)
train_c_y = to_categorical(train_c_y, 2)
val_c_y = to_categorical(val_c_y, 2)
test_c_y = to_categorical(test_c_y, 2)


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='classification_output_loss', factor=0.3, patience=10, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period1 = ModelCheckpoint(Name + '-{epoch:03d}-{val_segmentation_output_acc:.4f}.h5',
                                     monitor='classification_output_acc', mode='auto', save_best_only='True')

checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_segmentation_output_acc:.4f}.h5',
                                     monitor='segmentation_output_loss', mode='auto', period=20)

# earlyStopping = EarlyStopping(monitor='val_acc', patience=20, mode='auto')

# define loss and compile the model

print("------------------------------------------------ begin training ------------------------------------------------")
opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.01)

# model.compile(loss={'segmentation_output': focal_tversky, "classification_output": binary_crossentropy},
model.compile(loss={'segmentation_output': generalized_dice_loss, "classification_output": binary_crossentropy},
              loss_weights={'segmentation_output': 0.7, "classification_output": 0.3},
              optimizer=opt,
              metrics={'segmentation_output': ['accuracy', generalized_dice_coeff], "classification_output": ['accuracy']})


hist = model.fit([train_x, train_x],
                 [train_c_y, train_s_y],
                 batch_size=batch_size,
                 epochs=EPOCHS,
                 validation_data=(val_x, [val_c_y, val_s_y]),
                 verbose=1,
                 callbacks=[checkpoint_period2, checkpoint_period1, reduce_lr, csv_logger, reduce_lr],
                 shuffle=True)

print("------------------------------------------------ Saving model ------------------------------------------------")
model_filename = Name + ".h5"
model.save_weights(model_filename)
print('model saved to:', model_filename)
f = open('./output/' + Name + '.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
# plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["classification_output_loss"], label="classification_output_loss")
plt.plot(N, hist.history["segmentation_output_loss"], label="segmentation_output_loss")

plt.plot(N, hist.history["classification_output_acc"], label="classification_output_acc")
plt.plot(N, hist.history["segmentation_output_acc"], label="segmentation_output_acc")
plt.plot(N, hist.history["segmentation_output_dice_coef"], label="segmentation_output_dice_coef")

# plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.plot(N, hist.history["val_classification_output_loss"], label="val_classification_output_loss")
plt.plot(N, hist.history["val_segmentation_output_loss"], label="val_segmentation_output_loss")

plt.plot(N, hist.history["val_classification_output_acc"], label="val_classification_output_acc")
plt.plot(N, hist.history["val_segmentation_output_acc"], label="val_segmentation_output_acc")
plt.plot(N, hist.history["val_segmentation_output_dice_coef"], label="val_segmentation_output_dice_coef")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy/DSC")
plt.legend()
plt.savefig('./output/' + Name + '_plot.png')
print("------Start predicting------")


predictions_c, predictions_s = model.predict(test_x, batch_size=32)


print("------------------------------------------------ Classification testing ------------------------------------------------")

print(classification_report(test_c_y.argmax(axis=1), predictions_c.argmax(axis=1), target_names=lb.classes_, digits=4))

print("------------------------------------------------ Segmentation testing ------------------------------------------------")

# evaluate the model
# loss = model.evaluate(test_x, [test_c_y, test_s_y], verbose=0)
loss, cla_loss, seg_loss, cla_acc, seg_acc, seg_dice_coef = model.evaluate(test_x, [test_c_y, test_s_y], verbose=0)
print('Test total loss:', loss)
print('Test classification loss:', cla_loss)
print('Test segmentation loss:', seg_loss)

print('Test classification accuracy:', cla_acc)
print('Test segmentation accuracy:', seg_acc)
print('Test segmentation dice_coef:', seg_dice_coef)


preds_c, preds_s = model.predict(test_x, batch_size=8, verbose=1)
test_mask = test_s_y.flatten()
pred_mask = preds_s.flatten()
fpr, tpr, thresholds = roc_curve(test_mask, pred_mask, pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

Precision, Recall, accuracy, F1, IoU = p_r_f1_iou(test_s_y, preds_s)
print('Precision:', Precision)
print('Recall:', Recall)
print('m_Iou:', IoU)
print('F1_score:', F1)

