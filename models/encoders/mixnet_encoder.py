from keras.layers import *
from .keras_mixnets import MixNetSmall


class MixNetEncoder():
    def __init__(self):
        self.downsample_factor = 8
        # each model will have one down sample factor

    def __call__(self, input_tensor, *args, **kwargs):
        model = MixNetSmall(input_tensor=input_tensor)
        # model.summary()
        inner = model.layers[-137].output
        return inner, self.downsample_factor

if __name__ == '__main__':
    input_shape = (1024, 64, 3)
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    model = MixNetEncoder()
    inner, _ = model(inputs)
    print(inner)
