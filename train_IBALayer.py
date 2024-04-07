import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from keras.utils.np_utils import *
from sklearn.preprocessing import LabelBinarizer
from IBA.tensorflow_v2 import IBALayer, model_wo_softmax, to_saliency_map
warnings.filterwarnings("ignore")
import utils_paths
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import tqdm
import cv2
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
import tensorflow as tf


from IBA.utils import plot_saliency_map
from IBA.tensorflow_v1 import IBALayer, model_wo_softmax, to_saliency_map


def scheduler(epoch):
    if epoch % 200 == 0 and epoch != 0:
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

print("TensorFlow version: {}, Keras version: {}".format(
    tf.version.VERSION, keras.__version__))

batch_size = 12
num_classes = 2
epochs = 5
width = 64
height = 64
depth = 3
INIT_LR = 0.01
target = (width, height)
data_augmentation = True

Name = "VGG16_64X64_5"

train_path = './dataset/ICIS2/train/images/'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')

train_imagePaths = sorted(list(utils_paths.list_images(train_path)))
val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(307)
random.shuffle(train_imagePaths)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)

# The data, split between train and test sets:
train_x, train_y = load_data(train_imagePaths, target)
val_x, val_y = load_data(val_imagePaths, target)
test_x, test_y = load_data(test_imagePaths, target)

train_x = train_x.astype('float32')
val_x = val_x.astype('float32')
test_x = test_x.astype('float32')

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)
train_y = to_categorical(train_y, len(lb.classes_))
val_y = to_categorical(val_y, len(lb.classes_))
test_y = to_categorical(test_y, len(lb.classes_))

# set True to train the model yourself
run_training = True
# run_training = False
model_weight_url = "https://userpage.fu-berlin.de/leonsixt/cifar_weights.h5"

print('train_x shape:', train_x.shape)
print('val_x shape', val_x.shape)
print('test_x shape', test_x.shape)
# Convert class vectors to binary class matrices.

K.clear_session()
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', use_bias=True,
                 input_shape=train_x.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', use_bias=True))
# model.add(BatchNormalization(name='bn1'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# model.add(Conv2D(64, (5, 5), padding='same', name='conv2', use_bias=True))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', use_bias=True))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', use_bias=True))
model.add(BatchNormalization(name='bn2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))  # 8

# just add IBALayer where you want to explain the model.
# you could even add multiple IBALayers to a single model.

model.add(Conv2D(128, (5, 5), padding='same', name='conv3', use_bias=True))
model.add(BatchNormalization(name='bn3'))
model.add(Activation('relu', name='relu3'))

model.add(Conv2D(128, (5, 5), padding='same', name='conv4', use_bias=True))
model.add(Activation('relu', name='relu4'))
model.add(GlobalAveragePooling2D(name='pool4'))

model.add(Dropout(0.5, name='dropout1'))
model.add(Dense(1024, name='fc1'))
model.add(Activation('relu', name='relu5'))

model.add(Dropout(0.5, name='dropout2'))
model.add(Dense(num_classes, name='fc2'))
model.add(Activation('softmax', name='softmax'))

if not run_training:
    print("loading weights")
    model.load_weights('cifar_weights.h5')

opt = keras.optimizers.SGD(lr=INIT_LR, momentum=0.9, decay=1e-5, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Score trained model.
scores = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
predictions = model.predict(test_x, batch_size=12)
print(classification_report((test_y.argmax(axis=1)),
                            predictions.argmax(axis=1), target_names=lb.classes_, digits=3))  #


if run_training:
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
    )

    datagen.fit(train_x)

    # Fit the model on the batches generated by datagen.flow().
    # model.load_weights('23_300_weights.h5')
    # reduce_lr = LearningRateScheduler(scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-7, mode='auto', verbose=1)
    hist = model.fit_generator(
        datagen.flow(train_x, train_y, batch_size=batch_size),
        epochs=epochs,
        # initial_epoch=300,
        steps_per_epoch=len(train_x) // batch_size,
        validation_data=(val_x, val_y), workers=4, callbacks=[reduce_lr])

    # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    accy = hist.history['acc']
    lossy = hist.history['loss']
    np_accy = np.array(accy).reshape((1, len(accy)))
    np_lossy = np.array(lossy).reshape((1, len(lossy)))
    dataframe = pd.DataFrame({'train_Loss': np_lossy, 'training_accuracy': np_accy})
    dataframe.to_csv(Name + ".csv", index=True)  # 路径可以根据需要更改
    print("保存loss and ACC 日志文件成功")
    model_filename = Name + ".h5"
    model.save_weights(model_filename)
    print('model saved to:', model_filename)

# Score trained model.
scores = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
predictions = model.predict(test_x, batch_size=12)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_, digits=3))  # ,

model_logits = model_wo_softmax(model)
iba = model.layers[7]
target = iba.set_classification_loss(model_logits.output)

# ensure model is in eval mode
K.set_learning_phase(0)

# estimate mean, std on 5000 samples
for img in tqdm(train_x[:303]):
    iba.fit({model.input: img[None]})

rows = 2
cols = 6
fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))

for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
    img = test_x[i:i + 1]
    ax0.imshow(img[0])
    ax0.set_xticks([])
    ax0.set_yticks([])
    target = test_y[i].nonzero()[0]
    capacity = iba.analyze({model.input: img, iba.target: target})
    saliency_map = to_saliency_map(capacity, shape=(32, 32))

    plot_saliency_map(saliency_map, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)

## Access to internal values
# collect all intermediate tensors
iba.collect_all()

# storing all tensors can slow down the optimization.
# you can also select to store only specific ones:
# iba.collect("alpha", "model_loss")
# to only collect a subset all all tensors

# run analyze

i = 4
img = test_x[i][None]
target = test_y[i].nonzero()[0]
capacity = iba.analyze({model.input: img, iba.target: target})
saliency_map = to_saliency_map(capacity, shape=(32, 32))

# get all saved variables
report = iba.get_report()
print("iterations:", list(report.keys()))
print("{:<30} {:}".format("name:", "shape"))
print()
for name, val in report['init'].items():
    print("{:<30} {:}".format(name + ":", str(val.shape)))

# Losses during optimization
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].set_title("cross entrop loss")
ax[0].plot(list(report.keys()), [it['model_loss'] for it in report.values()])

ax[1].set_title("mean capacity")
ax[1].plot(list(report.keys()), [it['capacity_mean'] for it in report.values()])

# Distribution of alpha (pre-softmax) values per iteraton
cols = 6
rows = len(report) // cols

fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.2 * rows))

for ax, (it, values) in zip(axes.flatten(), report.items()):
    ax.hist(values['alpha'].flatten(), log=True, bins=20)
    ax.set_title("iteration: " + str(it))

plt.subplots_adjust(wspace=0.3, hspace=0.5)
fig.suptitle("distribution of alpha (pre-softmax) values per iteraton.", y=1)
plt.savefig('./output/alpha.png')

# Distributiuon of the final capacity
plt.hist(report['final']['capacity'].flatten(), bins=20, log=True)
plt.title("Distributiuon of the final capacity")
plt.show()
plt.savefig('./output/capacity.png')

#
# if run_training:
#     # datagen = ImageDataGenerator(
#     #     rotation_range=20,
#     #     width_shift_range=0.1,
#     #     height_shift_range=0.1,
#     #     horizontal_flip=True,  # randomly flip images
#     # )
#     datagen = ImageDataGenerator(
#         rotation_range=0.2,
#         width_shift_range=0.05,
#         height_shift_range=0.05,
#         shear_range=0.05,
#         zoom_range=0.05,
#         horizontal_flip=True,
#         fill_mode='nearest')
#     datagen.fit(train_x)
#
#     # Fit the model on the batches generated by datagen.flow().
#     hist = model.fit_generator(
#         datagen.flow(train_x, train_y, batch_size=batch_size),
#         epochs=epochs,
#         steps_per_epoch=len(train_x) // batch_size,
#         validation_data=(val_x, val_y), workers=4,
#         callbacks=[reduce_lr]
#     )
#     model.save_weights("cifar_weights.h5")
#
# # Score trained model.
# scores = model.evaluate(test_x, test_y, verbose=1)
# predictions = model.predict(test_x, batch_size=12)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
# print(classification_report(test_y.argmax(axis=1),
#     predictions.argmax(axis=1), target_names=lb.classes_))
#
# model_logits = model_wo_softmax(model)
# iba = model.layers[7]
# target = iba.set_classification_loss(model_logits.output)
#
# # ensure model is in eval mode
# K.set_learning_phase(0)
# # estimate mean, std on 5000 samples
# for img in tqdm(train_x[:303]):
#     iba.fit({model.input: img[None]})
#
#
# rows = 2
# cols = 6
# fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 2. * rows))
#
# for i, ax0, ax1 in zip(range(rows * cols // 2), ax.flatten()[::2], ax.flatten()[1::2]):
#     img = test_x[i:i + 1]
#     ax0.imshow(img[0])
#     ax0.set_xticks([])
#     ax0.set_yticks([])
#     target = test_y[i].nonzero()[0]
#     capacity = iba.analyze({model.input: img, iba.target: target},
#                            batch_size=batch_size,
#                            steps=10,  # steps (int): number of iterations to optimize
#                            beta=4.5,
#                            learning_rate=INIT_LR,
#                            min_std=0.05,
#                            smooth_std=0,
#                            normalize_beta=True,
#                            # session=None,
#                            # pass_mask=None,
#                            )
#     saliency_map = to_saliency_map(capacity, shape=(32, 32))
#
#     plot_saliency_map(saliency_map, ax=ax1, colorbar_size=0.15, colorbar_fontsize=10)
#
# ## Access to internal values
# # collect all intermediate tensors
# iba.collect_all()
#
# # storing all tensors can slow down the optimization.
# # you can also select to store only specific ones:
# # iba.collect("alpha", "model_loss")
# # to only collect a subset all all tensors
#
# # run analyze
# i = 4
# img = test_x[i][None]
# target = test_y[i].nonzero()[0]
# capacity = iba.analyze({model.input: img, iba.target: target})
# saliency_map = to_saliency_map(capacity, shape=(32, 32))
#
# # get all saved variables
# report = iba.get_report()
# print("iterations:", list(report.keys()))
# print("{:<30} {:}".format("name:", "shape"))
# print()
# for name, val in report['init'].items():
#     print("{:<30} {:}".format(name + ":", str(val.shape)))
#
#
# # Losses during optimization
# fig, ax = plt.subplots(1, 2, figsize=(8, 3))
# ax[0].set_title("cross entrop loss")
# ax[0].plot(list(report.keys()), [it['model_loss'] for it in report.values()])
#
# ax[1].set_title("mean capacity")
# ax[1].plot(list(report.keys()), [it['capacity_mean'] for it in report.values()])
#
#
# # Distribution of alpha (pre-softmax) values per iteraton
# cols = 6
# rows = len(report) // cols
#
# fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.2 * rows))
#
# for ax, (it, values) in zip(axes.flatten(), report.items()):
#     ax.hist(values['alpha'].flatten(), log=True, bins=20)
#     ax.set_title("iteration: " + str(it))
#
# plt.subplots_adjust(wspace=0.3, hspace=0.5)
#
# fig.suptitle("distribution of alpha (pre-softmax) values per iteraton.", y=1)
# plt.show()
#
#
# # Distributiuon of the final capacity
#
# plt.hist(report['final']['capacity'].flatten(), bins=20, log=True)
# plt.title("Distributiuon of the final capacity")
# plt.show()
