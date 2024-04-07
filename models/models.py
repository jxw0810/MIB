# from IBA.tensorflow_v1 import IBALayer
from keras.applications import VGG16, VGG19, ResNet50
from keras import Model, Sequential

# 导入所需模块
from keras.layers import *
from keras.initializers import TruncatedNormal
from keras import backend as K

class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # FC层
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # softmax 分类
        model.add(Dense(classes, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("softmax"))

        return model

class VGG16Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        for layer in model.layers:
            layer.trainable = True  # 涓嶈皟鏁翠箣鍓嶇殑鍗风Н灞傜殑鍙傛暟

        output = Flatten(name='flatten')(model.output)  # 鍘绘帀鍏ㄨ繛鎺ュ眰锛屽墠闈㈤兘鏄嵎绉眰
        output = Dense(256, activation='sigmoid', name='fc1')(output)
        output = Dropout(0.5)(output)
        output = Dense(classes, activation='softmax')(output)  # model灏辨槸鏈€鍚庣殑y
        vgg_model = Model(inputs=model.input, outputs=output, name='VGG16Net')

        return vgg_model
    
    
class VGG19Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model = VGG19(include_top=False, weights='imagenet', input_shape=inputShape)
        for layer in model.layers:
            layer.trainable = True  # 涓嶈皟鏁翠箣鍓嶇殑鍗风Н灞傜殑鍙傛暟

        output = Flatten(name='flatten')(model.output)  # 鍘绘帀鍏ㄨ繛鎺ュ眰锛屽墠闈㈤兘鏄嵎绉眰
        output = Dense(256, activation='sigmoid', name='fc1')(output)
        output = Dropout(0.5)(output)
        output = Dense(classes, activation='softmax')(output)  # model灏辨槸鏈€鍚庣殑y
        vgg_model = Model(inputs=model.input, outputs=output, name='VGG19Net')
        
        return vgg_model


class ResNet34Net_pre:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # 加载keras模型
        model = ResNet50(include_top=False, weights='imagenet', input_shape=inputShape)
        for layer in model.layers:
            layer.trainable = True  # 不调整之前的卷积层的参数

        output = Flatten(name='flatten')(model.get_layer('activation_34').output)  # 去掉全连接层，前面都是卷积层
        output = Dense(4096, activation='relu', name='fc1')(output)
        output = Dense(256, activation='sigmoid', name='fc2')(output)
        output = Dropout(0.5)(output)
        output = Dense(classes, activation='softmax')(output)  # model就是最后的y
        resnet34_model = Model(inputs=model.input, outputs=output, name='ResNet34Net')


        return resnet34_model

class ResNet50Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # 加载keras模型
        model = ResNet50(include_top=False, weights='imagenet', input_shape=inputShape)
        for layer in model.layers:
            layer.trainable = True  # 不调整之前的卷积层的参数

        # output = Flatten(name='flatten')(model.output)  # 去掉全连接层，前面都是卷积层
        output = Flatten(name='flatten')(model.output)  # 去掉全连接层，前面都是卷积层
        output = Dense(4096, activation='relu', name='fc1')(output)
        output = Dense(4096, activation='relu', name='fc2')(output)
        output = Dropout(0.5)(output)
        output = Dense(classes, activation='softmax')(output)  # model就是最后的y
        resnet50_model = Model(inputs=model.input, outputs=output, name='ResNet50Net')


        return resnet50_model

