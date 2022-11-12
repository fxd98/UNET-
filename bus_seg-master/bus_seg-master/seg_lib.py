import numpy as np

from keras import Model
from keras.models import Sequential
from keras.layers.core import Reshape, Flatten
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Dropout, concatenate
from keras.layers.merge import add, multiply, subtract
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, \
    UpSampling2D, Lambda
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, concatenate, \
    BatchNormalization, Concatenate


## FUNCTIONS

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))


def voe_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (1 - intersection /
            (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))


## LAYERS

def expend_as(x, n):
    y = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),#shape=(None, 112, 112, 32)
               arguments={'repnum': n})(x)

    return y


def conv_bn_act(x, filters, drop_out=0.0):
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def bn_act_conv_dense(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def dense_block(x, elements=3, filters=8, drop_out=0.0):
    blocks = [x]

    for i in range(elements):
        temp = bn_act_conv_dense(x, filters, drop_out)
        blocks.append(temp)
        x = Concatenate(axis=-1)(blocks)

    return x


def selective_layer(x, filters, compression=0.5, drop_out=0.0):
    x1 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')(x)#shape=(None, 224, 224, 16)

    if drop_out > 0:
        x1 = Dropout(drop_out)(x1)

    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)#shape=(None, 224, 224, 16)

    x2 = Conv2D(filters, (3, 3), padding='same')(x)#shape=(None, 224, 224, 16)

    if drop_out > 0:
        x2 = Dropout(drop_out)(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)#shape=(None, 224, 224, 16)

    x3 = add([x1, x2])#shape=(None, 224, 224, 16)

    x3 = GlobalAveragePooling2D()(x3)#shape=(None, 16)

    x3 = Dense(int(filters * compression))(x3)#它是一个全连接的网络层
    x3 = BatchNormalization()(x3)#shape=(None, 16)
    x3 = Activation('relu')(x3)

    x3 = Dense(filters)(x3)#shape=(None, 16)

    x3p = Activation('sigmoid')(x3)

    x3m = Lambda(lambda x: 1 - x)(x3p)

    x4 = multiply([x1, x3p])# shape=(None, 224, 224, 16)  相乘
    x5 = multiply([x2, x3m])#shape=(None, 224, 224, 16)

    return add([x4, x5])


def selective_transition_layer(x, filters, drop_out=0.0):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = selective_layer(x, filters, drop_out=drop_out)

    return x


def transition_layer(x, compression, drop_out=0.0):
    n = K.int_shape(x)[-1]

    n = int(n * compression)

    x = BatchNormalization()(x)
    x = Conv2D(n, (1, 1), padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    return x


def attention_layer(d, e, n):
    d1 = Conv2D(n, (1, 1), activation=None, padding='same')(d)#shape=(None, 112, 112, 1)

    e1 = Conv2D(n, (1, 1), activation=None, padding='same')(e)#shape=(None, 112, 112, 1)

    concat_de = add([d1, e1])#shape=(None, 112, 112, 1)

    relu_de = Activation('relu')(concat_de)
    conv_de = Conv2D(1, (1, 1), padding='same')(relu_de)#shape=(None, 112, 112, 1)
    sigmoid_de = Activation('sigmoid')(conv_de)

    shape_e = K.int_shape(e)#(None, 112, 112, 32)
    upsample_psi = expend_as(sigmoid_de, shape_e[3])

    return multiply([upsample_psi, e])#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致


## 分割网络

def unet(filters=16, dropout=0, size=(224, 224, 1), attention_gates=False):
    inp = Input(size)#224, 224, 1

    c1 = conv_bn_act(inp, filters)#224, 224, 16
    c1 = conv_bn_act(c1, filters)#224, 224, 16
    p1 = MaxPooling2D((2, 2))(c1)#112, 112, 16
    filters = 2 * filters

    c2 = conv_bn_act(p1, filters)#112, 112, 32
    c2 = conv_bn_act(c2, filters)#112, 112, 32
    p2 = MaxPooling2D((2, 2))(c2)#56, 56, 32
    filters = 2 * filters

    c3 = conv_bn_act(p2, filters)#56，56，64
    c3 = conv_bn_act(c3, filters)#56，56，64
    p3 = MaxPooling2D((2, 2))(c3)#28，28，64
    filters = 2 * filters

    c4 = conv_bn_act(p3, filters)
    c4 = conv_bn_act(c4, filters)#28, 28, 128
    p4 = MaxPooling2D((2, 2))(c4)#14, 14, 128
    filters = 2 * filters

    cm = conv_bn_act(p4, filters)
    cm = conv_bn_act(cm, filters)#14, 14, 512

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)#28, 28, 256

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)#28, 28, 384

    c5 = conv_bn_act(u4, filters)#28, 28, 256
    c5 = conv_bn_act(c5, filters)#28, 28, 256

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)#56, 56, 128

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)#56, 56, 192

    c6 = conv_bn_act(u3, filters)#56, 56, 128
    c6 = conv_bn_act(c6, filters)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)#112, 112, 64

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)#112, 112, 96

    c7 = conv_bn_act(u2, filters)
    c7 = conv_bn_act(c7, filters)#112, 112, 64

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)#224, 224, 32

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)

    else:

        u1 = concatenate([u1, c1], axis=3)#224, 224, 48

    c8 = conv_bn_act(u1, filters)
    c8 = conv_bn_act(c8, filters)#224, 224, 32

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)#224, 224, 1

    return Model(inputs=[inp], outputs=[c9])


def selective_unet(filters=16, drop_out=0, compression=0.5, size=(224, 224, 1),#shape=(None, 224, 224, 16)
                   half_net=False, attention_gates=False):
    inp = Input(size)#shape=(None, 224, 224, 1)

    c1 = selective_layer(inp, filters, compression=compression,#shape=(None, 224, 224, 16)
                         drop_out=drop_out)
    c1 = selective_layer(c1, filters, compression=compression,#shape=(None, 224, 224, 16)
                         drop_out=drop_out)
    p1 = MaxPooling2D((2, 2))(c1)# shape=(None, 112, 112, 16)
    filters = 2 * filters

    c2 = selective_layer(p1, filters, compression=compression,#shape=(None, 112, 112, 32)
                         drop_out=drop_out)
    c2 = selective_layer(c2, filters, compression=compression,#shape=(None, 112, 112, 32)
                         drop_out=drop_out)
    p2 = MaxPooling2D((2, 2))(c2)#shape=(None, 56, 56, 32) 
    filters = 2 * filters

    c3 = selective_layer(p2, filters, compression=compression,#shape=(None, 56, 56, 64)
                         drop_out=drop_out)
    c3 = selective_layer(c3, filters, compression=compression,#shape=(None, 56, 56, 64)
                         drop_out=drop_out)
    p3 = MaxPooling2D((2, 2))(c3)#shape=(None, 28, 28, 64)
    filters = 2 * filters

    c4 = selective_layer(p3, filters, compression=compression,#shape=(None, 28, 28, 128)
                         drop_out=drop_out)
    c4 = selective_layer(c4, filters, compression=compression,#shape=(None, 28, 28, 128)
                         drop_out=drop_out)
    p4 = MaxPooling2D((2, 2))(c4)#shape=(None, 14, 14, 128)
    filters = 2 * filters

    cm = selective_layer(p4, filters, compression=compression,#shape=(None, 14, 14, 256)
                         drop_out=drop_out)
    cm = selective_layer(cm, filters, compression=compression,#shape=(None, 14, 14, 256)
                         drop_out=drop_out)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)#shape=(None, 28, 28, 128)

    if attention_gates:

        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)

    else:

        u4 = concatenate([u4, c4], axis=3)#shape=(None, 28, 28, 256)

    if half_net:

        c5 = conv_bn_act(u4, filters, drop_out=drop_out)
        c5 = conv_bn_act(c5, filters, drop_out=drop_out)

    else:

        c5 = selective_layer(u4, filters, compression=compression,#shape=(None, 28, 28, 128)
                             drop_out=drop_out)
        c5 = selective_layer(c5, filters, compression=compression,#shape=(None, 28, 28, 128)
                             drop_out=drop_out)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)#shape=(None, 56, 56, 64)

    if attention_gates:

        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)

    else:

        u3 = concatenate([u3, c3], axis=3)#shape=(None, 56, 56, 128)

    if half_net:

        c6 = conv_bn_act(u3, filters, drop_out=drop_out)
        c6 = conv_bn_act(c6, filters, drop_out=drop_out)

    else:

        c6 = selective_layer(u3, filters, compression=compression,#shape=(None, 56, 56, 64)
                             drop_out=drop_out)
        c6 = selective_layer(c6, filters, compression=compression,#shape=(None, 56, 56, 64)
                             drop_out=drop_out)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)#shape=(None, 112, 112, 32)

    if attention_gates:

        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)

    else:

        u2 = concatenate([u2, c2], axis=3)#shape=(None, 112, 112, 64)

    if half_net:

        c7 = conv_bn_act(u2, filters, drop_out=drop_out)
        c7 = conv_bn_act(c7, filters, drop_out=drop_out)

    else:

        c7 = selective_layer(u2, filters, compression=compression,#shape=(None, 112, 112, 32)
                             drop_out=drop_out)
        c7 = selective_layer(c7, filters, compression=compression,#shape=(None, 112, 112, 32)
                             drop_out=drop_out)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)#shape=(None, 224, 224, 16)

    if attention_gates:

        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)# shape=(None, 224, 224, 16)

    else:

        u1 = concatenate([u1, c1], axis=3)#shape=(None, 224, 224, 32) 

    if half_net:

        c8 = conv_bn_act(u1, filters, drop_out=drop_out)
        c8 = conv_bn_act(c8, filters, drop_out=drop_out)

    else:

        c8 = selective_layer(u1, filters, compression=compression,#shape=(None, 224, 224, 16)
                             drop_out=drop_out)
        c8 = selective_layer(c8, filters, compression=compression,#shape=(None, 224, 224, 16)
                             drop_out=drop_out)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)#shape=(None, 224, 224, 1)

    return Model(inputs=[inp], outputs=[c9])


def dense_unet(filters=8, blocks=3, layers=3, compression=0.5, drop_out=0,
               size=(224, 224, 1), half_net=False, attention_gates=False):
    inp = Input(size)

    x = Conv2D(filters, (3, 3), activation=None, padding='same')(inp)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    names = {}

    for i in range(layers):
        x = dense_block(x, blocks, filters, drop_out)
        x = transition_layer(x, compression, drop_out)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        name = 'x' + str(i + 1)
        names[name] = x

        x = MaxPooling2D((2, 2))(x)

        filters = 2 * filters

    x = dense_block(x, blocks, filters, drop_out)
    x = transition_layer(x, compression, drop_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(layers):

        filters = filters // 2

        name = 'x' + str(layers - i)

        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

        if attention_gates:

            x = concatenate([x, attention_layer(x, names[name], 1)], axis=3)

        else:

            x = concatenate([x, names[name]], axis=3)

        if half_net:

            x = conv_bn_act(x, filters, drop_out)
            x = conv_bn_act(x, filters, drop_out)

        else:

            x = dense_block(x, blocks, filters, drop_out)
            x = transition_layer(x, compression, drop_out)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

    x = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(x)

    return Model(inputs=[inp], outputs=[x])
