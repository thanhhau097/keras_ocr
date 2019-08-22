from keras.layers import *
from keras.models import Model

from base.base_model import BaseModel
from models.decoders.attention_decoder import AttentionDecoder
from models.rnn_encoders.rnn_encoder import RNNEncoder
from models.visual_encoders.mobilenet_encoder import MobileNetEncoder
from utils.ocr_utils import ctc_lambda_func


class JointModel(BaseModel):
    def __init__(self, config):
        super(JointModel, self).__init__(config)
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
        print('after RNN encoder:', encoder_outputs)

        # we have 2 branch here, 1 for CTC and 1 for Attention
        # ------------------- CTC -------------------
        # input is encoder_output
        num_class = self.config.n_letters + 1 - 2  # (-2 because of start and end token)
        ctc_inner = Dense(num_class, kernel_initializer='he_normal', name='dense2')(encoder_outputs)
        y_pred_ctc = Activation('softmax', name='softmax')(ctc_inner)

        max_text_len = self.config.hyperparameter.max_text_len
        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')  # (None ,8)
        input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
        label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([y_pred_ctc, labels, input_length, label_length])

        # test function
        self.test_func = K.function([inputs, labels, input_length, label_length], [y_pred_ctc, loss_out])
        self.pred_func = K.function([inputs], [y_pred_ctc])

        # ----------------- Attention --------------------
        decoder = AttentionDecoder(self.config)
        attention_inner, decoder_inputs = decoder(encoder_outputs, state)
        print("After Attention Decoder:", attention_inner)

        y_pred_attention = attention_inner

        # --------- JOINT MODEL -----------
        self.model = Model(inputs=[inputs, labels, input_length, label_length, decoder_inputs],
                           output=[loss_out, y_pred_attention])
        lossWeights = {"ctc": 1.0, "attention": 1.0}
        self.model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred,
                                                   'attention': 'categorical_crossentropy'})
        self.model.summary()

    def get_downsample_factor(self):
        return self.downsample_factor


if __name__ == '__main__':
    from utils.config import process_config
    config = process_config('../configs/config.json')
    model = JointModel(config=config)
