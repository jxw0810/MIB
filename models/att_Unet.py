from keras.layers import *
from IBA.tensorflow_v1 import IBALayer
from keras import backend as K
from keras import Model, layers
from keras.applications import vgg16

def attention_up_and_concate(down_layer, layer, data_format):

    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = down_layer
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 2, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):

    # theta_x(?,g_height,g_width,inter_channel)
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)
    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)
    # att_x(?,x_height,x_width,x_channel)
    att_x = multiply([x, rate])

    return att_x


def att_UNet(input_width, input_height, nClasses):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(vgg_streamlined, Model)

    data_format = K.image_data_format()

    # 解码层
    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    # o = vgg_streamlined.output
    o = attention_up_and_concate(vgg_streamlined.get_layer(name="block4_pool").output, o, data_format=data_format)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = attention_up_and_concate(vgg_streamlined.get_layer(name="block3_pool").output, o, data_format=data_format)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = attention_up_and_concate(vgg_streamlined.get_layer(name="block2_pool").output, o, data_format=data_format)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = attention_up_and_concate(vgg_streamlined.get_layer(name="block1_pool").output, o, data_format=data_format)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
    # 此处直接上采样置原始大小
    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)
    o = Activation("softmax")(o)

    model = Model(inputs=img_input, outputs=o)
    return model
