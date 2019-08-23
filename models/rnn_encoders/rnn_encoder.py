from keras.layers import *
from keras.layers.merge import add, concatenate


class RNNEncoder(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, input_tensor, *args, **kwargs):
        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(256, return_sequences=True,
                         kernel_initializer='he_normal', name='gru1')(input_tensor)
        gru_1b = GRU(256, return_sequences=True,
                          go_backwards=True, kernel_initializer='he_normal',
                          name='gru1b')(input_tensor)
        gru1_merged = add([gru_1, gru_1b])
        gru_2, state_2 = GRU(256, return_sequences=True,
                                  kernel_initializer='he_normal',
                                  return_state=True, name='gru2')(gru1_merged)
        gru_2b, state_2b = GRU(256, return_sequences=True, go_backwards=True,
                                    kernel_initializer='he_normal',
                                    return_state=True, name='gru2b')(gru1_merged)

        # transforms RNN output to character activations:
        encoder_outputs = concatenate([gru_2, gru_2b])
        state = concatenate([state_2, state_2b])

        return encoder_outputs, state
