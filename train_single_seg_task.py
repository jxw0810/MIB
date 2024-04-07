from gernerate_data import split_train_and_test, load_SEG_data, load_SEG_data_for_test
from sklearn.metrics import classification_report, auc, roc_curve
from models.Unet import UNet, UNet_IBA
from models.att_Unet import att_UNet
from models.r2_unet import r2_UNet
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import sgd, Adam
from utils.losses import dice_coef_loss, dice_coef, AUC, focal_tversky
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2 as cv
import numpy as np
import os


INIT_LR = 1e-3
EPOCHS = 200
BS = 16
width = 224
height = 224
depth = 3
target = (width, height)
shuffle = True
num_classes = 2
Name = 'UNet'
# Name = 'UNet_IBA_DESm'
preds_savePath = "./seg_prediction/BUSI/UNet-IB"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


tf.reset_default_graph()

model = UNet(width, height, num_classes)
# model = att_UNet(width, height, num_classes)
# model = UNet_IBA(width, height, num_classes)
# model = r2_UNet(width, height, num_classes)
# weight_path = "UNet_IBA_DESm-031-0.9417.h5"
# model.load_weights(weight_path)
model.summary()

trainX_dir = 'dataset/Dataset_BUSI_AN/train/images/'
# trainX_dir = 'dataset/DESm/train/images/'
# trainX_dir = 'dataset/DESm/train/images/'
trainY_dir = trainX_dir.replace('images', 'masks')

valX_dir = trainX_dir.replace('train', 'val')
valY_dir = valX_dir.replace('images', 'masks')

testX_dir = trainX_dir.replace('train', 'test')
testY_dir = testX_dir.replace('images', 'masks')

train_X, train_Y, _ = load_SEG_data(trainX_dir, target=target, shuffle=True)
#(train_X, val_X, train_Y, val_Y) = train_test_split(train_X, train_Y, test_size=0.2)
val_X, val_Y, _ = load_SEG_data(valX_dir, target=target, shuffle=True)
test_X, test_Y, test_img_Names, test_orignal_images = load_SEG_data_for_test(testX_dir, target=target, shuffle=False)

print("train_X shape:", train_X.shape)
print("train_X shape:", train_Y.shape)
print("val_X shape:", val_X.shape)
print("val_Y shape:", val_Y.shape)
print("test_X shape:", test_X.shape)
print("test_Y shape:", test_Y.shape)


# define callbacks
csv_logger = CSVLogger(Name+'.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-8, mode='auto', verbose=1)

checkpoint_period1 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
                                     monitor='val_acc', mode='auto', save_best_only='True')

# checkpoint_period2 = ModelCheckpoint(Name + '-{epoch:03d}-{val_acc:.4f}.h5',
#                                      monitor='val_acc', mode='auto', period=20)

# model.compile(loss=[tversky_dice_loss], optimizer=sgd, metrics=['accuracy', dice_coef])
# model.compile(optimizer=Adam(lr=INIT_LR), loss=binary_crossentropy, metrics=['accuracy', dice_coef, AUC])
model.compile(optimizer=Adam(lr=INIT_LR), loss=[focal_tversky], metrics=['accuracy', dice_coef])
history = model.fit(train_X, train_Y,
                    # initial_epoch=60,
                    batch_size=BS,
                    epochs=EPOCHS,
                    validation_data=(val_X, val_Y),
                    verbose=1,
                    callbacks=[checkpoint_period1, checkpoint_period2, reduce_lr, csv_logger],
                    shuffle=True)
                    
# save the model
model.save(os.path.join('./wights/', Name + '.h5'))

# evaluate the model
loss, acc, dice_coef = model.evaluate(test_X, test_Y, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)
print('Test dice_coef:', dice_coef)

# test the model and gernerate results 
preds = model.predict(test_X, batch_size=4)


preds_s = model.predict(test_X, batch_size=8, verbose=1)
test_mask = test_Y.flatten()
pred_mask = preds_s.flatten()
fpr, tpr, thresholds = roc_curve(test_mask, pred_mask, pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)


# def mask2gray(mask, input_type=None):
#     if input_type is 'pred':
#         mask0 = mask[:, :, 0]
#         mask1 = mask[:, :, 1]
#         mask = mask1 - mask0
#         print("mask.shape: ", mask.shape)
#         #mask = np.argmax(mask, axis=-1)
#         # print( mask[mask > 0.5] )
#         mask[mask > 0] = 255
#         mask[mask <= 0] = 0
#     mask = mask.astype(dtype=np.uint8)
#     rst = mask.copy()
#     cv.normalize(mask, rst, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
#     return rst


# img_size = (width, height)
# for i in range(test_X.shape[0]):
#     img = test_orignal_images[i]
#
#     gt = test_Y[i, :, :, 1]
#     gt = mask2gray(gt)
#
#     pred = preds[i]
#     prediction = mask2gray(pred, input_type='pred')
#
#     save_img = np.zeros([img_size[0], img_size[1]*3, 3], dtype=np.uint8)
#     save_img[:, 0:img_size[0], :] = img[:, :, ::-1]
#     save_img[:, img_size[0]:img_size[1] * 2, :] = cv.cvtColor(gt, cv.COLOR_GRAY2RGB)
#     save_img[:, img_size[0] * 2:img_size[1] * 3, :] = cv.cvtColor(prediction, cv.COLOR_GRAY2RGB)
#
#     savePath = os.path.join(preds_savePath, "joint")
#
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)
#
#     cv.imwrite(savePath + "{0}".format(test_img_Names[i]), save_img)
#
# for i in range(test_X.shape[0]):
#
#     img = test_orignal_images[i]
#
#     gt = test_Y[i, :, :, 1]
#     gt = mask2gray(gt)
#     gt = cv.cvtColor(gt, cv.COLOR_GRAY2RGB)
#
#     pred_mask = preds[i]
#     pred_mask = mask2gray(pred_mask, input_type='pred_mask')
#
#     savePath = os.path.join(preds_savePath, "Imgs/")
#     saveMaskPath = os.path.join(preds_savePath, "Mask/")
#     savePredPath = os.path.join(preds_savePath, "Pred/")
#
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)
#     if not os.path.exists(saveMaskPath):
#         os.makedirs(saveMaskPath)
#     if not os.path.exists(savePredPath):
#         os.makedirs(savePredPath)
#
#     cv.imwrite(os.path.join(savePath, "{0}".format(test_img_Names[i])), img)
#     cv.imwrite(os.path.join(saveMaskPath, "{0}".format(test_img_Names[i])), gt)
#     cv.imwrite(os.path.join(savePredPath, "{0}".format(test_img_Names[i])), pred)
