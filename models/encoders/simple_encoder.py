from keras.layers import *
from keras.models import Model


class SimpleEncoder():
    def __init__(self):
        self.downsample_factor = 4
        # each model will have one down sample factor

    def __call__(self, input_tensor, *args, **kwargs):
        inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(
            input_tensor)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

        inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(
            inner)  # (None, 64, 32, 128)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

        inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(
            inner)  # (None, 32, 16, 256)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

        inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(
            inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

        inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(
            inner)  # (None, 32, 4, 512)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)

        return inner, self.downsample_factor