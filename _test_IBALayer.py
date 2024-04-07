import warnings
warnings.filterwarnings("ignore")
from keras.optimizers import SGD
from gernerate_data import load_cla_data
from keras.utils.np_utils import *
from sklearn.preprocessing import LabelBinarizer
from IBA.tensorflow_v2 import IBALayer, model_wo_softmax, to_saliency_map
from models.models import VGG16Net
import utils_paths
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import tqdm
import cv2 as cv
import keras
import keras.backend as K
import os
import tensorflow as tf
from IBA.utils import plot_saliency_map
from IBA.tensorflow_v1 import IBALayer, model_wo_softmax, to_saliency_map
from models.VGG import VGG16Net_IBA

batch_size = 12
num_classes = 2
EPOCHS = 200
width = 224
height = 224
depth = 3
INIT_LR = 0.01
target = (width, height)
data_augmentation = True
Name = "VGG16_BUSI"
GPU = True
if GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# train_path = 'dataset/test/images/'
train_path = './dataset/Dataset_BUSI/train/images/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')


train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
# The data, split between train and test sets:
train_x, train_y, _ = load_cla_data(train_imagePaths, target)
# val_x, val_y, _ = load_cla_data(val_imagePaths, target)
test_imagePaths = list(utils_paths.list_images(test_path))
# random.shuffle(test_imagePaths)
test_x, test_y, test_images_names = load_cla_data(test_imagePaths, target)

train_x = train_x.astype('float32')
# val_x = val_x.astype('float32')
test_x = test_x.astype('float32')

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
# val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)
# train_y = to_categorical(train_y, 2)
# val_y = to_categorical(val_y, 2)
# test_y = to_categorical(test_y, 2)
# Convert class vectors to binary class matrices.

# set True to train the model yourself
run_training = True

print('train_x shape:', train_x.shape)
# print('val_x shape', val_x.shape)
print('test_x shape', test_x.shape)


model = VGG16Net_IBA.build(width=width, height=height, depth=depth, classes=len(lb.classes_))
model.summary()
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/EPOCHS, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("loading weights")
# model.load_weights('wights/DES_VGG-IB-0.892-0.892.h5')
model.load_weights('wights/BUSI_VGGIB_0.909_0.905.h5')

# Score trained model.
print("------ evaluating ------")
scores = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("------ testing ------")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
      predictions.argmax(axis=1),
      target_names=lb.classes_,
      digits=3))


model_logits = model_wo_softmax(model)
iba = model.layers[19]
target = iba.set_classification_loss(model_logits.output)

# ensure model is in eval mode
K.set_learning_phase(0)

# estimate mean, std on 5000 samples
for img in train_x[:600]:
    iba.fit({model.input: img[None]})

rows = 6
cols = 2
fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
savePath = 'output/saliency map/LE/'
if not os.path.exists(savePath):
    os.makedirs(savePath)
# for i in range(test_x.shape[0]):
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i:i + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()
#
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
# savePath = 'output/saliency map/LE/'
# if not os.path.exists(savePath):
#     os.makedirs(savePath)
# # for i in range(test_x.shape[0]):
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i+6:i +6 + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i+6].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i+6])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()
#
#
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
# # for i in range(test_x.shape[0]):
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i+12:i +12 + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i+12].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i+12])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()
#
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
# # for i in range(test_x.shape[0]):
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i+18:i +18 + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i+18].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i+18])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()
#
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
#
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i+24:i +24 + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i+24].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i+24])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()
#
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
#
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#         img = test_x[i+30:i +30 + 1]
#         ax0.imshow(img[0])
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         target = test_y[i+30].nonzero()[0]
#         capacity = iba.analyze({model.input: img, iba.target: target})
#         saliency_map = to_saliency_map(capacity, shape=(224, 224))
#         img = np.squeeze(img)
#         save_img_Path = os.path.join(savePath, test_images_names[i+30])
#         plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#  # show all subplot after drawing
# plt.show()

fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))

for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
        img = test_x[i+36:i +36 + 1]
        ax0.imshow(img[0])
        ax0.set_xticks([])
        ax0.set_yticks([])
        target = test_y[i+36].nonzero()[0]
        capacity = iba.analyze({model.input: img, iba.target: target})
        saliency_map = to_saliency_map(capacity, shape=(224, 224))
        img = np.squeeze(img)
        save_img_Path = os.path.join(savePath, test_images_names[i+36])
        plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
 # show all subplot after drawing
plt.show()

fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))

for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
        img = test_x[i+42:i +42 + 1]
        ax0.imshow(img[0])
        ax0.set_xticks([])
        ax0.set_yticks([])
        target = test_y[i+42].nonzero()[0]
        capacity = iba.analyze({model.input: img, iba.target: target})
        saliency_map = to_saliency_map(capacity, shape=(224, 224))
        img = np.squeeze(img)
        save_img_Path = os.path.join(savePath, test_images_names[i+42])
        plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
 # show all subplot after drawing
plt.show()

fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))

for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
        img = test_x[i+48:i +48 + 1]
        ax0.imshow(img[0])
        ax0.set_xticks([])
        ax0.set_yticks([])
        target = test_y[i+48].nonzero()[0]
        capacity = iba.analyze({model.input: img, iba.target: target})
        saliency_map = to_saliency_map(capacity, shape=(224, 224))
        img = np.squeeze(img)
        save_img_Path = os.path.join(savePath, test_images_names[i+48])
        plot_saliency_map(saliency_map, img=img, savepath=save_img_Path, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
 # show all subplot after drawing
plt.show()
# Access to internal values
# collect all intermediate tensors
iba.collect_all()

# storing all tensors can slow down the optimization.
# you can also select to store only specific ones:
# iba.collect("alpha", "model_loss")
# to only collect a subset all all tensors

# run analyze
'''
i=4
img = test_x[i][None]
target = test_y[i].nonzero()[0]
capacity = iba.analyze({model.input: img, iba.target: target})
saliency_map = to_saliency_map(capacity, shape=(224, 224))

# get all saved variables
report = iba.get_report()
print("iterations:", list(report.keys()))
print("{:<30} {:}".format("name:", "shape"))
print()
for name, val in report['init'].items():
    print("{:<30} {:}".format(name + ":", str(val.shape)))

# Losses during optimization
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].set_title("cross entropy loss")
ax[0].plot(list(report.keys()), [it['model_loss'] for it in report.values()])
ax[1].set_title("mean capacity")
ax[1].plot(list(report.keys()), [it['capacity_mean'] for it in report.values()])

# Distribution of alpha (pre-softmax) values per iteration
cols = 6
rows = len(report) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.2 * rows))
for ax, (it, values) in zip(axes.flatten(), report.items()):
    ax.hist(values['alpha'].flatten(), log=True, bins=20)
    ax.set_title("iteration: " + str(it))

plt.subplots_adjust(wspace=0.3, hspace=0.5)
fig.suptitle("distribution of alpha (pre-softmax) values per iteration.", y=1)
plt.savefig('./output/alpha.png')

# Distribution of the final capacity
plt.hist(report['final']['capacity'].flatten(), bins=20, log=True)
plt.title("Distribution of the final capacity")
# plt.show()
plt.savefig('./output/capacity.png')
'''