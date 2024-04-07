from keras import Model
from keras.applications import vgg16
from keras.layers import *
from keras import backend as K

# 多任务网络
def Multi_task_VGG16(nClasses, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    base_model = vgg16.VGG16(
        # include_top=False,
        weights='imagenet',
        input_tensor=img_input
    )

    backbone = Model(inputs=base_model.input,
                     outputs=base_model.get_layer('block4_pool').output)


    # segmentation branch
    o = UpSampling2D((2, 2))(backbone.output)
    o = concatenate([backbone.get_layer(
        name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(
        name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(
        name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    shared_output = BatchNormalization()(o)

    # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
    # 此处直接上采样置原始大小
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)
    o = Activation("softmax")(o)

    s_model = Model(inputs=img_input, outputs=o)

    # classification branch

    c = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(shared_output)
    c = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(c)
    c = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(c)
    c = MaxPooling2D((2, 2), strides=(2, 2)(c))

    c = Flatten()(c)
    c = Dense(4096, activation='relu')(c)
    c = Dense(1024, activation='relu')(c)
    c = Dense(nClasses, activation='softmax')(c)

    c_model = Model(inputs=img_input, outputs=c)

    return s_model, c_model


class VGG_UNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        assert height % 32 == 0
        assert width % 32 == 0

        # 编码层
        inputs = Input(shape=inputShape)
        vgg_streamlined = vgg16.VGG16(
            include_top=False,
            weights='imagenet', input_tensor=inputs)
        assert isinstance(vgg_streamlined, Model)

        # 解码层
        o = UpSampling2D((2, 2))(vgg_streamlined.output)
        o = concatenate([vgg_streamlined.get_layer(
            name="block4_pool").output, o], axis=-1)
        o = Conv2D(512, (3, 3), padding="same")(o)
        o = BatchNormalization()(o)

        o = UpSampling2D((2, 2))(o)
        o = concatenate([vgg_streamlined.get_layer(
            name="block3_pool").output, o], axis=-1)
        o = Conv2D(256, (3, 3), padding="same")(o)
        o = BatchNormalization()(o)

        o = UpSampling2D((2, 2))(o)
        o = concatenate([vgg_streamlined.get_layer(
            name="block2_pool").output, o], axis=-1)
        o = Conv2D(128, (3, 3), padding="same")(o)
        o = BatchNormalization()(o)

        o = UpSampling2D((2, 2))(o)
        o = concatenate([vgg_streamlined.get_layer(
            name="block1_pool").output, o], axis=-1)
        o = Conv2D(64, (3, 3), padding="same")(o)
        o = BatchNormalization()(o)

        # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
        # 此处直接上采样置原始大小
        o = UpSampling2D((2, 2))(o)
        o = Conv2D(64, (3, 3), padding="same")(o)
        o = BatchNormalization()(o)

        o = Conv2D(classes, (1, 1), padding="same")(o)
        o = BatchNormalization()(o)
        o = Activation("relu")(o)

        # o = Reshape((-1, nClasses))(o)
        o = Activation("softmax")(o)

        model = Model(inputs=inputShape, outputs=o)
        return model


# Define the neural network
class unet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        inputs = Input(shape=inputShape)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        #
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        #
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

        up1 = UpSampling2D(size=(2, 2))(conv3)
        up1 = concatenate([conv2, up1], axis=-1)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
        #
        up2 = UpSampling2D(size=(2, 2))(conv4)
        up2 = concatenate([conv1, up2], axis=-1)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
        #  注意，写成这种结构，并且用的loss为categorical_crossentropy，
        # 需要对groundtruth数据进行处理，见后面help_function.py里的mask_Unet
        conv6 = Conv2D(classes, (1, 1), activation='relu', padding='same')(conv5)
        conv6 = core.Reshape((classes, height * width))(conv6)
        conv6 = core.Permute((2, 1))(conv6)
        #
        conv7 = core.Activation('softmax')(conv6)

        model = Model(inputs=inputShape, outputs=conv7)

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
        # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


# 可以修改模型的深度
class Unet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        data_format = 'channels_last'

        # if K.image_data_format() == "channels_first":
        #     inputShape = (depth, height, width)
        inputs = Input(shape=inputShape)
        x = inputs
        assert height % 32 == 0
        assert width % 32 == 0
        model_depth = 4
        features = 64
        skips = []
        for i in range(model_depth):
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            skips.append(x)
            x = MaxPooling2D((2, 2), data_format=data_format)(x)
            features = features * 2
        # 通道增加到1024
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

        for i in reversed(range(model_depth)):
            features = features // 2
            # attention_up_and_concate(x,[skips[i])
            x = UpSampling2D(size=(2, 2), data_format=data_format)(x)
            x = concatenate([skips[i], x], axis=-1)
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

        conv6 = Conv2D(classes, (1, 1), padding='same', data_format=data_format)(x)
        conv7 = core.Activation('sigmoid')(conv6)
        model = Model(inputs=inputs, outputs=conv7)

        # model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
        return model

