from keras.layers import Input, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, Activation, concatenate, Dense, Flatten
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Model

def binary_crossentropy(y_true, y_pred):
    e=1.0
    return K.mean(-(y_true*K.log(y_pred+K.epsilon())+
                    e*(1-y_true)*K.log(1-y_pred+K.epsilon())),
                  axis=-1)


def create_pair_model(input_width, input_height, depth, nClasses):
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


    # seg net
    x_2 = shared1(seg_input)
    x_2 = shared2(x_2)
    x_2 = shared3(x_2)
    x_2 = shared4(x_2)
    x_2 = shared5(x_2)
    block_1_out = shared6(x_2)
    x_2 = shared7(x_2)
    x_2 = shared8(x_2)
    x_2 = shared9(x_2)
    x_2 = shared10(x_2)
    x_2 = shared11(x_2)
    x_2 = shared12(x_2)
    block_2_out = shared13(x_2)
    x_2 = shared14(x_2)
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

    # Block 5
    x_2 = Conv2D(512, (3, 3), padding='same', name='s_block5_conv1')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)

    x_2 = Conv2D(512, (3, 3), padding='same', name='s_block5_conv2')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)

    x_2 = Conv2D(512, (3, 3), padding='same', name='sblock5_conv3')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)
    


    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x_2)
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


    # Block 5
    cls = Conv2D(512, (3, 3), padding='same', name='c_block5_conv1')(x_1)
    cls = BatchNormalization()(cls)
    cls = Activation('relu')(cls)

    cls = Conv2D(512, (3, 3), padding='same', name='c_block5_conv2')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('relu')(cls)

    cls = Conv2D(512, (3, 3), padding='same', name='c_block5_conv3')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('relu')(cls)
    
    cls = Flatten(name='cls_flatten')(cls)
    cls = Dense(64, name='cls_dense_0')(cls)
    cls = Activation('relu', name='cls_act_3')(cls)
    cls = Dropout(0.5, name='cls_dropout')(cls)
    cls = Dense(2, name='cls_dense_out')(cls)
    out_cls = Activation('sigmoid', name='classification_output')(cls)
    



    model = Model(inputs=[cls_input, seg_input], outputs=[out_cls, out_seg])


    return model