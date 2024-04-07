from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid, tanh
from keras import Model
# from models.Unet import UNet_IBA, UNet
from keras.applications import vgg16
import tensorflow as tf
from keras.layers import *
import math


def attach_attention_module(x, attention_module):
    
    if attention_module == 'se_block': # SE_block
        x, attention = se_block(x)
    elif attention_module == 'cbam_block': # CBAM_block
        x, attention = cbam_block(x)
    elif attention_module == 'gct_block': # GCT_block
        x, attention = gct_block(x)
    elif attention_module == 'eca_block': # GCT_block
        x, attention = eca_block(x)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return x, attention

def eca_block(inputs_tensor,num=None,gamma=2,b=1,**kwargs):

    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels,2)+b)/gamma))
    k = t if t%2 else t+1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels,1))(x_global_avg_pool)
    x = Conv1D(1,kernel_size=k,padding="same",name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)  #shape=[batch,chnnels,1]
    attention = Reshape((1, 1, channels))(x)
    output = multiply([inputs_tensor,attention])
    
    return output, attention


def gct(inputs_tensor, name=None, epsilon=1e-5, use_l2_norm=True, after_relu=False):

    _, width, height, num_channels = inputs_tensor.get_shape().as_list()
    channel_index = 3

    squeeze1 = [1, 2]
    squeeze2 = [3]
    param_size = [1, 1, 1, num_channels]

    alpha = 1.0
    gamma = 0.
    beta = 0.

    X = inputs_tensor

    if use_l2_norm:  
        # use l2 norm
        X_sqrt = K.square(X)
        l2_norm = K.sum(X_sqrt, squeeze1, keepdims=True)
        l2_norm = K.sqrt(l2_norm + epsilon)

        embedding = alpha * l2_norm

        mid = tf.reduce_mean(K.square(embedding), squeeze2, keep_dims=True)

        _embedding = embedding * (gamma / tf.sqrt(mid + epsilon)) + beta
    else:  
        # use l1 norm
        if after_relu:  # if after ReLU, all the values of X should be not negative.
            l1_norm = tf.reduce_sum(X, squeeze1, keep_dims=True)
        else:
            l1_norm = tf.reduce_sum(K.abs(X), squeeze1, keep_dims=True)

        embedding = alpha * l1_norm
        mid = tf.reduce_mean(K.abs(embedding), squeeze2, keep_dims=True)
        _embedding = embedding * (gamma / (mid + epsilon)) + beta

    gate = 1. + tanh(_embedding)

    X_ = gate * inputs_tensor
    

    return [X_, gate]


def gct_block(inputs_tensor):
    
    y = Lambda(lambda x: gct(x))(inputs_tensor)
    x, gate = y[0], y[1]
    return x, gate                    
                           
                           
                           
                           
def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                activation='relu',
                kernel_initializer='he_normal',
                use_bias=True,
                bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=True,
                bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1,1,channel)

    output = multiply([input_feature, se_feature])
    return output,output 

def cbam_block(cbam_feature, ratio=8):

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature, cbam_feature

def channel_attention(input_feature, ratio=8):

    channel_axis = -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                    activation='relu',
                    kernel_initializer='he_normal',
                    use_bias=True,
                    bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7


    channel = input_feature._keras_shape[-1]
    cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                kernel_size=kernel_size,
                padding='same',
                strides=1,
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)(concat)	
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])


def MTL_Attention_model(input_width, input_height, depth, nClasses, ):
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
#     iba = IBALayer()(x)
    
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv1')(x)
    c = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv2')(c)
    block4_c_conv3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_c_conv3')(c)

    c, attention_map = attach_attention_module(block4_c_conv3, 'eca_block')
    c = BatchNormalization()(c)
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
    o = add([block4_s_conv3, block4_c_conv3])
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Conv2D(256, (1, 1), padding='same', activation='relu')(o)
    o = Multiply()([o, attention_map])
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    block4_s_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_s_pool')(block4_s_conv3)
    
    
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
    s1 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='segmentation_output')(o)
    s2 = Conv2D(nClasses, (3, 3), activation='softmax', padding='same',name='seg-out')(o)


    return Model(inputs=clas_input, outputs=[c, s1, s2], name='Multi_task_VGG16')
