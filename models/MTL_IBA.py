from keras import Model
from models.models import VGG16Net, ResNet34Net_pre, ResNet50Net
# from models.Unet import UNet_IBA, UNet
from keras.applications import vgg16
from keras.layers import *
from keras.models import load_model
from IBA.tensorflow_v1 import IBALayer
from keras import backend as K



def MTL_IBA(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
#     base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

#     for layer in base_model.layers:
#         layer.trainable = True
#     # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    iba = IBALayer()
    layer1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')
    layer2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')
    layer3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')
    layer4 = BatchNormalization()
    layer5 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')
    # block 5
    layer6 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')
    layer7 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')
    layer8 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')
    layer9 = BatchNormalization()
    layer10 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')
    
    # classification branch
    
    c = x
    c = iba(c)
    c = layer1(c)
    c = layer2(c)
    c = layer3(c)
    c = layer4(c)
    c = layer5(c)
    c = layer6(c)
    c = layer7(c)
    c = layer8(c)
    c = layer9(c)
    c = layer10(c)
    
    c = Flatten(name='cls_flatten')(c)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    # segmentation branch
    o = x
    o = layer1(o)
    o = layer2(o)
    block4_conv3 = layer3(o)
#     block4_conv3 = layer4(o)
    o = layer5(block4_conv3)
    o = layer6(o)
    o = layer7(o)
    o = layer8(o)
#     o = layer9(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')



def MTL_IBA_h3(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
#     base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

#     for layer in base_model.layers:
#         layer.trainable = True
#     # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    # classification branch 
    # classification block 4
    c = IBALayer()(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(c)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)
    c = BatchNormalization()(c)
    c = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')(c)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')(c)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')(c)
    c = BatchNormalization()(x)
    c = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')(c)
     
    c = Flatten(name='cls_flatten')(c)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    # segmentation branch
    # segmentation block 4
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    o = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(block4_s_conv3)
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')


def MTL_IBA_cross(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
#     base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

#     for layer in base_model.layers:
#         layer.trainable = True
#     # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    # classification branch 
    # classification block 4
#     iba = IBALayer()(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    block4_c_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)
    iba = IBALayer()(block4_c_conv3)
    c = BatchNormalization()(iba)
    block4_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')(block4_c_pool)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')(c)
    block5_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')(c)
    c = BatchNormalization()(block5_c_conv3)
    block5_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')(c)
     
    c = Flatten(name='cls_flatten')(block5_c_pool)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    # segmentation branch
    # segmentation block 4
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    # cross1
    o = Multiply()([iba, block4_s_conv3])
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(256, (1, 1), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(o)
    
    
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(block4_s_pool)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    block5_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    # cross2
    o = Multiply()([block5_s_conv3, block5_c_conv3])
    o = concatenate([block5_s_conv3, o], axis=-1)
    o = Conv2D(512, (1, 1), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')


def MTL_IBA_cross2(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
#     base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

#     for layer in base_model.layers:
#         layer.trainable = True
#     # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    # classification branch 
    # classification block 4
#     iba = IBALayer()(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    block4_c_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)
    iba = IBALayer()(block4_c_conv3)
    c = BatchNormalization()(iba)
    block4_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')(block4_c_pool)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')(c)
    block5_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')(c)
    c = BatchNormalization()(block5_c_conv3)
    block5_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')(c)
     
    c = Flatten(name='cls_flatten')(block5_c_pool)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    # segmentation branch
    # segmentation block 4
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    # cross1
    o = Multiply()([iba, block4_s_conv3])
#     o = concatenate([block4_s_conv3, o, block4_c_conv3], axis=-1)
    o = add([block4_s_conv3, o, block4_c_conv3])
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(256, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(256, (3, 3), padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(o)
    
    
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(block4_s_pool)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    block5_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    # cross2
    o = Multiply()([block5_s_conv3, block5_c_conv3])
    o = add([block5_s_conv3, o])
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')


       

def MTL_IBA_cross3(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)
      

    # classification branch 
    # classification block 4
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    block4_c_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)
    iba = IBALayer(name='iba')(block4_c_conv3)
#     print(iba)
    c = BatchNormalization()(iba)
    block4_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')(block4_c_pool)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')(c)
    block5_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')(c)
    c = BatchNormalization()(block5_c_conv3)
    block5_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')(c)
     
    c = Flatten(name='cls_flatten')(block5_c_pool)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    clas_branch = Model(inputs=clas_input, outputs=c, name='clas_branch')
#     clas_branch.load_weights("MTL_IBA_cross3_pretrain.h5", by_name=True)
#     for layer in clas_branch.layers:
#         layer.trainable = False
    
    
    # segmentation branch
    # segmentation block 4
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    # cross1    
    o = add([block4_s_conv3, o, block4_c_conv3])
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Multiply()([o, iba])
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(o)
    
    
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(block4_s_pool)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    block5_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    # cross2
    o = Multiply()([block5_s_conv3, block5_c_conv3])
    o = concatenate([block5_s_conv3, o], axis=-1)
    o = Conv2D(512, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(512, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(512, (1, 1), padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(block5_s_conv3)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(2, (3, 3), activation='sigmoid', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(2, (3, 3), activation='sigmoid', padding='same',name='seg-out')(o)

        # shared feature extraction module: backbone
#     seg_branch = Model(inputs=clas_input, outputs=x)
#     for layer in seg_branch.layers:
#         layer.trainable = True
    
    
    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')

def MTL_IBA_cross3_pretrain(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    # classification branch 
    # classification block 4
    
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(c)
    block4_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(c)
    iba = IBALayer(name='iba')(block4_c_conv3)
    c = BatchNormalization()(iba)
    block4_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(block4_c_pool)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(c)
    block5_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3')(c)
    c = BatchNormalization()(block5_c_conv3)
    block5_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(c)
     
    c = Flatten(name='cls_flatten')(block5_c_pool)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
#     clas_branch = Model(inputs=clas_input, outputs=block5_c_pool, name='clas_branch')
# #     clas_branch.load_weights("MTL_IBA_cross3_pretrain.h5", by_name=True)
#     for layer in clas_branch.layers:
#             layer.trainable = False
    
    # segmentation branch
    # segmentation block 4
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    # cross1
    o = Multiply()([iba, block4_s_conv3])
#     o = concatenate([block4_s_conv3, o, block4_c_conv3], axis=-1)
    o = add([block4_s_conv3, o, block4_c_conv3])
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(o)
    
    
    
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(block4_s_pool)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    block5_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    # cross2
    o = Multiply()([block5_s_conv3, block5_c_conv3])
    o = add([block5_s_conv3, o])
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(512, (3, 3), padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='sigmoid', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)

    
    return Model(inputs=clas_input, outputs=[c, s1, s2], name='MTL_IBA_cross3_pretrain')

def MTL_IBA_s2c(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

#     iba = IBALayer()(x)

    # segmentation block 4
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv1')(x)
    o = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv2')(o)
    block4_s_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_s_conv3')(o)
    # cross1
    iba = IBALayer()(block4_s_conv3)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(o)
    
    # segmentation block 5
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv1')(block4_s_pool)
    o = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv2')(o)
    block5_s_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_s_conv3')(o)
    
    # UP 1
    o = UpSampling2D((2, 2))(o)
    o = concatenate([block4_s_conv3, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 2
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 3
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    # UP 4
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    
    
    # classification branch 
    # classification block 4
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    block4_c_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)
    # cross1
    c = add([block4_s_conv3, block4_c_conv3])
    c = Conv2D(256, (1, 1), padding='same', activation='relu')(c)
    c = Conv2D(256, (1, 1), padding='same', activation='relu')(c)
    c = Conv2D(256, (1, 1), padding='same', activation='relu')(c)
    c = Multiply()([c, iba])
    c = BatchNormalization()(c)
    c = Activation('relu')(c)
    block4_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')(c)
    
    # classification block 5
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')(block4_c_pool)
    c = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')(c)
    block5_c_conv3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')(c) 
    # cross2
    c = Multiply()([block5_c_conv3, block5_s_conv3])
    c = concatenate([block5_c_conv3, c], axis=-1)
    c = Conv2D(512, (1, 1), padding='same', activation='relu')(c)
    c = Conv2D(512, (1, 1), padding='same', activation='relu')(c)
    c = Conv2D(512, (1, 1), padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = Activation('relu')(c)
    block5_c_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')(c)
     
    c = Flatten(name='cls_flatten')(block5_c_pool)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('sigmoid', name='classification_output')(c)
    
    
    # segmentation branch
    # last conv
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='MTL_IBA_s2c')