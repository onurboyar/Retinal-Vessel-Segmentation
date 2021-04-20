from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

from utils.metrics import dice_coef, jacard


def up_and_concate(down_layer, layer):

    up = UpSampling2D(size=(2, 2))(down_layer)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])

    return concate


def attention_up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x, rate])

    return att_x

def AttentionUNet(input_size = (608,576,1)):
    inputs = Input(input_size)
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i])
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    conv6 = Conv2D(1, (1, 1), padding='same')(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy',dice_coef,jacard, tf.keras.metrics.AUC(), tf.keras.metrics.MeanIoU(num_classes=2),
                                                                                        tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model
