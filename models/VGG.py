from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.applications.vgg16 import VGG16
from IBA.tensorflow_v1 import IBALayer

class VGG16Net_IBA:
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
            layer.trainable = True
            
#         def gct_iba(inputs_tensor):
#             output = Lambda(lambda x: IBALayer()(x))(inputs_tensor)
#             y, score_map = output[0], output[1]
#             return y, score_map 
#         output = IBALayer()(model.output)
        
        output = IBALayer()(model.output)
        output = Flatten(name='flatten')(output)
        output = Dense(1024, activation='relu', name='fc1')(output)
        output = Dense(256, activation='sigmoid', name='fc2')(output)
        output = Dropout(0.5)(output)
        output = Dense(classes, activation='softmax')(output)
        vgg_model = Model(inputs=model.input, outputs=output, name='vgg16')

        return vgg_model

class VGG16Net_IBA3:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        model = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        
        
        for layer in model.layers:
            layer.trainable = True
        base_model = Model(inputs=model.input, outputs=model.get_layer('block3_pool').output, name='vgg16')
        
        # Block 4
        output = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(base_model.output)
        output = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(output)
        output = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(output)
        output = IBALayer()(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(output)

        # Block 5
        output = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(output)
        output = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(output)
        output = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(output)
        output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(output)

        
        output = Flatten(name='flatten')(output)
        output = Dense(64, activation='relu', name='fc1')(output)
        output = Dropout(0.5)(output)
        output = Dense(2, activation='softmax')(output)
        vgg_model = Model(inputs=base_model.input, outputs=output, name='vgg16')

        return vgg_model