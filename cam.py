from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import plot_model
from keras import Model
from models.VGG import VGG16Net_IBA
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
import cv2


def load_model_h5(model_file):
    return load_model(model_file)


def load_img_preprocess(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  #
    img = preprocess_input(img)  #
    return img


def gradient_compute(model, layername, img):
    preds = model.predict(img)
    idx = np.argmax(preds[0])  #

    output = model.output[:, idx]  #
    last_layer = model.get_layer(layername)

    grads = K.gradients(output, last_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))  #
    #
    iterate = K.function([model.input], [pooled_grads, last_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([img])

    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    return conv_layer_output_value


def plot_heatmap(conv_layer_output_value, img_in_path, img_out_path):
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_in_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimopsed_img = heatmap * 0.4
    superimopsed_img = heatmap * 0.4 + img

    cv2.imwrite(img_out_path, superimopsed_img)

# dataset/BUSI_M/test/images/malignant_images/malignant (65).png
img_path = "dataset/BUSI_M/test/images/malignant_images/malignant (123).png"
model_path = "wights/BUSI_VGGIB_0.909_0.905.h5"
img_out_path = 'output/BUSI/malignant (123)_2.png'
layername = 'block5_pool'
# layername = 'IBALayer'
img = load_img_preprocess(img_path, (224, 224))
# model = load_model_h5(model_path)

model = VGG16Net_IBA.build(width=224, height=224, depth=3, classes=3)
model.summary()
model.load_weights(model_path)

conv_value = gradient_compute(model, layername, img)
plot_heatmap(conv_value, img_path, img_out_path)
