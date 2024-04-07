from keras.layers import *
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Model

def binary_crossentropy(y_true, y_pred):
    e=1.0
    return K.mean(-(y_true*K.log(y_pred+K.epsilon())+
                    e*(1-y_true)*K.log(1-y_pred+K.epsilon())),
                  axis=-1)


def create_pair_model(input_width, input_height, depth, nClasses):
    
#     img_input = Input(shape=(input_height, input_width, 3), name='input')
    cls_input = Input(shape=(input_height, input_width, 3), name='cls_input')
    seg_input = Input(shape=(input_height, input_width, 3), name='seg_input')
    # assume cls_input is the same as seg_input
    # shared layers
    # block 1
    shared1 = Conv2D(64, (3, 3), padding='same', name='share_conv1_1')
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
    shared44 = MaxPooling2D()

    # cls net
    x_1 = shared1(cls_input)
    x_1 = shared2(x_1)
    x_1 = shared3(x_1)
    x_1 = shared4(x_1)
    x_1 = shared5(x_1)
    block_1_out = shared6(x_1)
    block_1_pool_out = shared7(block_1_out)
    x_1 = shared8(block_1_pool_out)
    x_1 = shared9(x_1)
    x_1 = shared10(x_1)
    x_1 = shared11(x_1)
    x_1 = shared12(x_1)
    block_2_out = shared13(x_1)
    block_2_pool_out = shared14(block_2_out)
    x_1 = shared15(block_2_pool_out)
    x_1 = shared16(x_1)
    x_1 = shared17(x_1)
    x_1 = shared18(x_1)
    x_1 = shared19(x_1)
    x_1 = shared20(x_1)
    x_1 = shared21(x_1)
    x_1 = shared22(x_1)
    block_3_out = shared23(x_1)
    block_3_pool_out = shared24(block_3_out)
    x_1 = shared25(block_3_pool_out)
    x_1 = shared26(x_1)
    x_1 = shared27(x_1)
    x_1 = shared28(x_1)
    x_1 = shared29(x_1)
    x_1 = shared30(x_1)
    x_1 = shared31(x_1)
    x_1 = shared32(x_1)
    block_4_out = shared33(x_1)
    block_4_pool_out = shared34(block_4_out)
    x_1 = shared35(block_4_pool_out)
    x_1 = shared36(x_1)
    x_1 = shared37(x_1)
    x_1 = shared38(x_1)
    x_1 = shared39(x_1)
    x_1 = shared40(x_1)
    x_1 = shared41(x_1)
    x_1 = shared42(x_1)
    block_5_out = shared43(x_1)
    block_5_pool_out = shared44(block_5_out)

    conv1_cls = block_5_pool_out
#     conv1_seg = block_5_pool_out
#     c = Dense(1024, activation='relu', name='fc1')(c)
#     c = Dense(256, activation='sigmoid', name='fc2')(c)
#     c = Dropout(0.5)(c)
#     c = Dense(nClasses, activation='softmax', name='classification_output')(c)
    
    x = Flatten(name='cls_flatten')(conv1_cls)
    x = Dense(64, name='cls_dense_0')(x)
    x = Activation('relu', name='cls_act_3')(x)
    x = Dropout(0.5, name='cls_dropout')(x)
    x = Dense(2, name='cls_dense_out')(x)
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
    out_seg2 = Conv2D(2, (3, 3), activation='softmax', padding='same', name='seg-out')(x)
    # out_seg = Activation(activation='softmax', name='seg-out')(conv10)

    model = Model(inputs=[cls_input,seg_input], outputs=[out_cls, out_seg, out_seg2])
    


#     # UP 1
#     # segmentation branch
#     o = conv1_seg
#     o = UpSampling2D((2, 2))(o)
#     o = concatenate([block_4_pool_out, o], axis=-1)  # [(None, 56, 56, 512), (None, 112, 112, 512)]
#     o = Conv2D(256, (3, 3), padding="same")(o)
#     o = BatchNormalization()(o)

#     o = UpSampling2D((2, 2))(o)
#     o = concatenate([block_3_pool_out, o], axis=-1)
#     o = Conv2D(256, (3, 3), padding="same")(o)
#     o = BatchNormalization()(o)

#     o = UpSampling2D((2, 2))(o)
#     o = concatenate([block_2_pool_out, o], axis=-1)
#     o = Conv2D(128, (3, 3), padding="same")(o)
#     o = BatchNormalization()(o)

#     o = UpSampling2D((2, 2))(o)
#     o = concatenate([block_1_pool_out, o], axis=-1)
#     o = Conv2D(64, (3, 3), padding="same")(o)
#     o = BatchNormalization()(o)

#     o = UpSampling2D((2, 2))(o)
#     o = Conv2D(64, (3, 3), padding="same")(o)
#     o = BatchNormalization()(o)

#     o = Conv2D(nClasses, (1, 1), padding="same")(o)
#     o = BatchNormalization()(o)
#     o = Activation("relu")(o)
#     out_seg = Activation("softmax", name='segmentation_output')(o)
#     out_seg2 = Activation("softmax", name='seg_output')(o)

#     model = Model(inputs=a_input, outputs=[out_cls, out_seg, out_seg2])


    return model