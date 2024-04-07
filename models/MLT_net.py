from keras import Model
from models.models import VGG16Net, ResNet34Net_pre, ResNet50Net
# from models.Unet import UNet_IBA, UNet
from keras.applications import vgg16
from keras.layers import *
from keras.models import load_model
# from IBA.tensorflow_v1 import IBALayer
from keras import backend as K

    

def MTL_classic(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    # img_input = Input(shape=(input_height, input_width, depth))
    base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

    for layer in base_model.layers:
        layer.trainable = True
    # shared feature extraction module: backbone
    backbone = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    # classification branch
    c = backbone.output
    c = Flatten(name='cls_flatten')(c)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
    c = Activation('relu', name='cls_act_4')(c)
    c = Activation('sigmoid', name='classification_output')(c)

    # segmentation branch
    o = backbone.get_layer(name="block5_conv3").output
    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block4_conv3").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block3_conv3").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block2_conv2").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([backbone.get_layer(name="block1_conv2").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = Activation("relu")(o)
    s = Activation("softmax", name='segmentation_output')(o)

    return Model(inputs=backbone.inputs, outputs=[c, s])


def MTL_classic2(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    
    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(clas_input)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    block3_conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)
    
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

    # classification branch
    c = x
    c = Flatten(name='cls_flatten')(c)
    c = Dense(64, name='cls_dense_0')(c)
    c = Activation('relu', name='cls_act_3')(c)
    c = Dropout(0.5, name='cls_dropout')(c)
    c = Dense(2, name='cls_dense_out')(c)
#     c = Activation('relu', name='cls_act_4')(c)
    c = Activation('sigmoid', name='classification_output')(c)

    

    # UP 1
    o = block5_conv3
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
    s = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)

    return Model(inputs=[clas_input,seg_input], outputs=[c, s], name='Multi_task_VGG16')


def MTL_3(input_width, input_height, depth, nClasses):
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
    c = layer1(c)
    c = layer2(c)
    c = layer3(c)
#     c = layer4(c)
    c = layer5(c)
    c = layer6(c)
    c = layer7(c)
    c = layer8(c)
#     c = layer9(c)
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
#    o = layer4(block4_conv3)
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


def MTL_3_2(input_width, input_height, depth, nClasses):
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

def MTL_3_2_IBA(input_width, input_height, depth, nClasses):
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



def MTL3_interaction(input_width, input_height, depth, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    clas_input = Input(shape=(input_height, input_width, depth))
    seg_input = Input(shape=(input_height, input_width, depth))
    
    # Block 1 
    x = Conv2D(64, (3, 3),padding='same', name='block1_conv1')(clas_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    
#     base_model = VGG16Net.build(width=input_width, height=input_height, depth=depth, classes=nClasses)

#     for layer in base_model.layers:
#         layer.trainable = True
#     # shared feature extraction module: backbone
    backbone = Model(inputs=clas_input, outputs=x)

    layer1 = Conv2D(512, (3, 3), padding='same', name='block4_conv1')
    layer2 = BatchNormalization()
    layer3 = Activation('relu')
    layer4 = Conv2D(512, (3, 3), padding='same', name='block4_conv2')
    layer5 = BatchNormalization()
    layer6 = Activation('relu')
    layer7 = Conv2D(512, (3, 3), padding='same', name='block4_conv3')
    layer8 = BatchNormalization()
    layer9 = Activation('relu')
    layer10 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
    # block 5
    layer11 = Conv2D(512, (3, 3), padding='same', name='block5_conv1')
    layer12 = BatchNormalization()
    layer13 = Activation('relu')
    layer14 = Conv2D(512, (3, 3), padding='same', name='block5_conv2')
    layer15 = BatchNormalization()
    layer16 = Activation('relu')
    layer17 = Conv2D(512, (3, 3), padding='same', name='block5_conv3')
    layer18 = BatchNormalization()
    layer19 = Activation('relu')
    layer20 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
    
    # classification branch
    
    c = x
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
    c = layer11(c)
    c = layer12(c)
    c = layer13(c)
    c = layer14(c)
    c = layer15(c)
    c = layer16(c)
    c = layer17(c)
    c = layer18(c)
#     c = layer19(c)
    c = layer20(c)

    
#     layer10 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')

#     layer1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')

#     layer2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')

#     layer3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')
#     layer4 = BatchNormalization()
#     layer5 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_c_pool')
#     # block 5
#     layer6 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv1')

#     layer7 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv2')
#     x = BatchNormalization()(x)
#     layer8 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_c_conv3')
#     layer9 = BatchNormalization()
#     layer10 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_c_pool')
    
    
#     # classification branch
    
#     c = x
#     c = layer1(c)
#     c = layer2(c)
#     c = layer3(c)
# #     c = layer4(c)
#     c = layer5(c)
#     c = layer6(c)
#     c = layer7(c)
#     c = layer8(c)
# #     c = layer9(c)
#     c = layer10(c)
    
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
    o = layer3(o)
    o = layer4(o)
    o = layer5(o)
    o = layer6(o)
    o = layer7(o)
    o = layer8(o)
    block4_conv3 = layer9(o)
    o = layer10(block4_conv3)
    o = layer11(o)
    o = layer12(o)
    o = layer13(o)
    o = layer14(o)
    o = layer15(o)
    o = layer16(o)
    o = layer17(o)
    o = layer18(o)
    o = layer19(o)
    
#     o = x
#     o = layer1(o)
#     o = layer2(o)
#     block4_conv3 = layer3(o)
# #     block4_conv3_pool = layer4(block4_conv3)
#     o = layer5(block4_conv3)
#     o = layer6(o)
#     o = layer7(o)
#     o = layer8(o)
# #     o = layer9(o)
    
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


def create_pair_model(input_width, input_height, depth, nClasses):
    cls_input = Input(shape=(input_width, input_height, 3), name='cls_input')
    seg_input = Input(shape=(input_width, input_height, 3), name='seg_input')
    # assume cls_input is the same as seg_input
    # shared layers
    # block 1
    shared1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1')
    shared2 = BatchNormalization()
    shared3 = Activation('relu')
    shared4 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')
    shared5 = BatchNormalization()

    # unet block 1 output
    shared6 = Activation('relu')
    shared7 = MaxPooling2D()

    # block 2
    shared8 = Conv2D(128, (3, 3), padding='same', name='block2_conv1')
    shared9 = BatchNormalization()
    shared10 = Activation('relu')

    shared11 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')
    shared12 = BatchNormalization()
    shared13 = Activation('relu')

    shared14 = MaxPooling2D()

    # block 3
    shared15 = Conv2D(256, (3, 3), padding='same', name='block3_conv1')
    shared16 = BatchNormalization()
    shared17 = Activation('relu')

    shared18 = Conv2D(256, (3, 3), padding='same', name='block3_conv2')
    shared19 = BatchNormalization()
    shared20 = Activation('relu')

    shared21 = Conv2D(256, (3, 3), padding='same', name='block3_conv3')
    shared22 = BatchNormalization()
    shared23 = Activation('relu')

    shared24 = MaxPooling2D()

    # block 4
    shared25 = Conv2D(512, (3, 3), padding='same', name='block4_conv1')
    shared26 = BatchNormalization()
    shared27 = Activation('relu')

    shared28 = Conv2D(512, (3, 3), padding='same', name='block4_conv2')
    shared29 = BatchNormalization()
    shared30 = Activation('relu')

    shared31 = Conv2D(512, (3, 3), padding='same', name='block4_conv3')
    shared32 = BatchNormalization()
    shared33 = Activation('relu')

    shared34 = MaxPooling2D()

    # Block 5
    shared35 = Conv2D(512, (3, 3), padding='same', name='block5_conv1')
    shared36 = BatchNormalization()
    shared37 = Activation('relu')

    shared38 = Conv2D(512, (3, 3), padding='same', name='block5_conv2')
    shared39 = BatchNormalization()
    shared40 = Activation('relu')

    shared41 = Conv2D(512, (3, 3), padding='same', name='block5_conv3')
    shared42 = BatchNormalization()
    shared43 = Activation('relu')

    # cls net
    x_1 = shared1(cls_input)
    x_1 = shared2(x_1)
    x_1 = shared3(x_1)
    x_1 = shared4(x_1)
    x_1 = shared5(x_1)
    x_1 = shared6(x_1)
    x_1 = shared7(x_1)
    x_1 = shared8(x_1)
    x_1 = shared9(x_1)
    x_1 = shared10(x_1)
    x_1 = shared11(x_1)
    x_1 = shared12(x_1)
    x_1 = shared13(x_1)
    x_1 = shared14(x_1)
    x_1 = shared15(x_1)
    x_1 = shared16(x_1)
    x_1 = shared17(x_1)
    x_1 = shared18(x_1)
    x_1 = shared19(x_1)
    x_1 = shared20(x_1)
    x_1 = shared21(x_1)
    x_1 = shared22(x_1)
    x_1 = shared23(x_1)
    x_1 = shared24(x_1)
    x_1 = shared25(x_1)
    x_1 = shared26(x_1)
    x_1 = shared27(x_1)
    x_1 = shared28(x_1)
    x_1 = shared29(x_1)
    x_1 = shared30(x_1)
    x_1 = shared31(x_1)
    x_1 = shared32(x_1)
    x_1 = shared33(x_1)
    x_1 = shared34(x_1)
    x_1 = shared35(x_1)
    x_1 = shared36(x_1)
    x_1 = shared37(x_1)
    x_1 = shared38(x_1)
    x_1 = shared39(x_1)
    x_1 = shared40(x_1)
    x_1 = shared41(x_1)
    x_1 = shared42(x_1)
    x_1 = shared43(x_1)

    conv1_cls = x_1

    x = Flatten(name='flatten')(conv1_cls)
    x = Dense(64, name='dense_0')(x)
    x = Activation('relu', name='cls_act_0')(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(2, name='dense_1')(x)
    out_cls = Activation('sigmoid', name='classification_output')(x)

    # seg net
    x_2 = shared1(seg_input)
    x_2 = shared2(x_2)
    x_2 = shared3(x_2)
    x_2 = shared4(x_2)
    x_2 = shared5(x_2)
    block_1_out = shared6(x_2)
    x_2 = shared7(block_1_out)
    x_2 = shared8(x_2)
    x_2 = shared9(x_2)
    x_2 = shared10(x_2)
    x_2 = shared11(x_2)
    x_2 = shared12(x_2)
    block_2_out = shared13(x_2)
    x_2 = shared14(block_2_out)
    x_2 = shared15(x_2)
    x_2 = shared16(x_2)
    x_2 = shared17(x_2)
    x_2 = shared18(x_2)
    x_2 = shared19(x_2)
    x_2 = shared20(x_2)
    x_2 = shared21(x_2)
    x_2 = shared22(x_2)
    block_3_out = shared23(x_2)
    x_2 = shared24(block_3_out)
    x_2 = shared25(x_2)
    x_2 = shared26(x_2)
    x_2 = shared27(x_2)
    x_2 = shared28(x_2)
    x_2 = shared29(x_2)
    x_2 = shared30(x_2)
    x_2 = shared31(x_2)
    x_2 = shared32(x_2)
    block_4_out = shared33(x_2)
    x_2 = shared34(block_4_out)
    x_2 = shared35(x_2)
    x_2 = shared36(x_2)
    x_2 = shared37(x_2)
    x_2 = shared38(x_2)
    x_2 = shared39(x_2)
    x_2 = shared40(x_2)
    x_2 = shared41(x_2)
    x_2 = shared42(x_2)
    x_2 = shared43(x_2)
    conv1_seg = x_2


    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv1_seg)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    out_seg = Conv2D(2, (3, 3), activation='softmax', padding='same', name='segmentation_output')(x)
    # out_seg = Activation(activation='softmax', name='seg-out')(conv10)

    model = Model(inputs=[cls_input, seg_input], outputs=[out_cls, out_seg])


    return model
