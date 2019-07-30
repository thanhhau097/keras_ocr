from keras.layers import *
from keras.layers.merge import add, concatenate


class SimpleDecoder():
    def __init__(self):
        pass

    def __call__(self, input_tensor, *args, **kwargs):
        # RNN layer
        # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
        #     input_tensor)  # (None, 32, 512)
        # print(lstm_1)
        # lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
        #     input_tensor)
        # reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
        #
        # lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
        # lstm1_merged = BatchNormalization()(lstm1_merged)
        #
        # lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        # lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        #     lstm1_merged)
        # reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)
        #
        # lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)
        # lstm2_merged = BatchNormalization()(lstm2_merged)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(512, return_sequences=True,
                    kernel_initializer='he_normal', name='gru1')(input_tensor)
        gru_1b = GRU(512, return_sequences=True,
                     go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(input_tensor)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(512, return_sequences=True,
                    kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(512, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        inner = concatenate([gru_2, gru_2b])

        return inner