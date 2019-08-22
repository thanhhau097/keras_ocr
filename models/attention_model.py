from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from base.base_model import BaseModel
from models.decoders.attention_decoder import AttentionDecoder
from models.rnn_encoders.rnn_encoder import RNNEncoder
from models.visual_encoders.mobilenet_encoder import MobileNetEncoder


class AttentionModel(BaseModel):
    def __init__(self, config):
        super(AttentionModel, self).__init__(config)
        self.config = config
        self.build_model()

    def build_model(self):
        input_shape = (None, 64, 3)

        # Make Network
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

        # ENCODER
        encoder = MobileNetEncoder()
        inner, self.downsample_factor = encoder(inputs)
        print("After Encoder:", inner)

        # CNN to RNN
        shape = inner.shape
        target_shape = (-1, int(shape[2] * shape[3]))
        inner = Reshape(target_shape=target_shape, name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)
        # inner = Permute((2, 1, 3), name='permute')(inner)
        # inner = TimeDistributed(Flatten(), name='timedistrib')(inner)
        print("After CNN to RNN:", inner)

        # RNN Encoder
        rnn_encoder = RNNEncoder(self.config)
        encoder_outputs, state = rnn_encoder(inner)

        # DECODER
        decoder = AttentionDecoder(self.config)
        inner, decoder_inputs = decoder(encoder_outputs, state)
        print("After Decoder:", inner)

        y_pred = inner
        # self.test_func = K.function([inputs, decoder_inputs], [y_pred])
        # self.pred_func = K.function([inputs, decoder_inputs], [y_pred])

        self.model = Model([inputs, decoder_inputs], y_pred)
        # https://arxiv.org/pdf/1904.08364.pdf
        optimizer = Adam(0.2, decay=0.5)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        self.model.summary()

    def get_downsample_factor(self):
        return self.downsample_factor


if __name__ == '__main__':
    from utils.config import process_config
    config = process_config('../configs/config.json')
    model = AttentionModel(config=config)
