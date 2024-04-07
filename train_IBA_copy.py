# Tensorflow: IBACopy
import warnings
warnings.filterwarnings("ignore")
from IBA.tensorflow_v1 import IBACopy, TFWelfordEstimator, model_wo_softmax, to_saliency_map
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, label_binarize
from keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.metrics import classification_report
from utils.load_and_save_Model import load_data
from IBA.utils import plot_saliency_map
from keras.applications import VGG16
from models.VGG import VGG16Net_IBA
from keras.models import load_model
from keras.utils.np_utils import *
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import keras.backend as K
from PIL import Image
import utils_paths
import random
import keras


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

print("TensorFlow version: {}, Keras version: {}".format(
    tf.version.VERSION, keras.__version__))


# data loading
def get_imagenet_generator(val_dir, image_size=(224, 224), shuffle=True, batch_size=50):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    return image_generator.flow_from_directory(
        val_dir, shuffle=shuffle, seed=0, batch_size=batch_size, target_size=image_size)


def norm_image(x):
    return (x - x.min()) / (x.max() - x.min())

BS = 12
width = 224
height = 224
depth = 3
target = (width, height)

train_path = '.\\dataset\\ICIS2\\train\\images\\'
test_path = train_path.replace('train', 'test')
val_path = train_path.replace('train', 'val')


val_imagePaths = sorted(list(utils_paths.list_images(val_path)))
test_imagePaths = sorted(list(utils_paths.list_images(test_path)))
random.seed(307)
random.shuffle(val_imagePaths)
random.shuffle(test_imagePaths)

# 遍历训练读取数据
val_x, val_y = load_data(val_imagePaths, target)
test_x, test_y = load_data(test_imagePaths, target)

# 数据集切分
# (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

# 转换标签为one-hot encoding格式
lb = LabelBinarizer()
val_y = lb.fit_transform(val_y)
test_y = lb.fit_transform(test_y)
val_y = to_categorical(val_y, 2)
test_y = to_categorical(test_y, 2)

img_batch, target_batch = next(get_imagenet_generator(val_path))

monkey_pil = Image.open("E:\\Multi-task IBA\\dataset\ICIS2\\test\\images\\malignant_images\\CESM_20_4A_CC_R_.jpg")
monkey_pil.resize((224, 224))
monkey = preprocess_input(np.array(monkey_pil))[None]
monkey_target = 0  # 382: squirrel monkey
monkey_target = to_categorical(monkey_target, 2)

# load model
model_softmax = load_model('weights_VGG16_ICIS_4-100-0.8919.h5')
# model_softmax = VGG16(weights='imagenet')

# remove the final softmax layer
model = model_wo_softmax(model_softmax)

# select layer after which the bottleneck will be inserted
feat_layer = model.get_layer(name='block4_conv1')
iba = IBACopy(feat_layer.output, model.output)

# checks if all variables are equal in the original graph and the copied graph
iba.assert_variables_equal()

iba_logits = iba.predict({model.input: val_x})
model_logits = model.predict(val_x)
assert (np.abs(iba_logits - model_logits).mean()) < 1e-5
print(np.abs(iba_logits - model_logits).mean())

feed_dict_gen = map(lambda x: {model.input: x[0]},
                    get_imagenet_generator(val_path))

iba.fit_generator(feed_dict_gen, n_samples=37)
print("Fitted estimator on {} samples".format(iba._estimator.n_samples()))

# set classification loss
target = iba.set_classification_loss()

# you can also specificy your own loss with :
# iba.set_model_loss(my_loss)

iba.set_default(beta=10)

# get the saliency map
capacity = iba.analyze(
    feature_feed_dict={model.input: monkey},
    copy_feed_dict={target: np.array([monkey_target])}
)
saliency_map = to_saliency_map(capacity, shape=(224, 224))
K.image_data_format()

plot_saliency_map(saliency_map, img=norm_image(monkey[0]))

# collect all intermediate tensors
iba.collect_all()

# storing all tensors can slow down the optimization.
# you can also select to store only specific ones:
# iba.collect("alpha", "model_loss")
# to only collect a subset all all tensors


# run the optimization
capacity = iba.analyze(
    feature_feed_dict={model.input: monkey},
    copy_feed_dict={iba.target: np.array([monkey_target])}
)

# get all saved outputs
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
plt.show()

# Distributiuon of the final capacity

plt.hist(report['final']['capacity'].flatten(), bins=20, log=True)
plt.title("Distributiuon of the final capacity")
plt.show()
