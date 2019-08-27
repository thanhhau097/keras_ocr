from keras.layers import *
from keras.models import Model
from models.visual_encoders.mobilenet_encoder import MobileNetEncoder

from base.base_model import BaseModel
from models.decoders.simple_decoder import SimpleDecoder
from utils.ocr_utils import ctc_lambda_func


class CTCModel(BaseModel):
    def __init__(self, config):
        super(CTCModel, self).__init__(config)
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

        # DECODER
        decoder = SimpleDecoder()
        inner = decoder(inner)
        print("After Decoder:", inner)

        # transforms RNN output to character activations:
        num_classes = self.config.n_letters + 1
        inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(inner)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        max_text_len = self.config.hyperparameter.max_text_len
        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')  # (None ,8)
        input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
        label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([y_pred, labels, input_length, label_length])

        # test function
        self.test_func = K.function([inputs, labels, input_length, label_length], [y_pred, loss_out])
        self.pred_func = K.function([inputs], [y_pred])

        # if training:
        self.model = Model(inputs=[inputs, labels, input_length, label_length],
                           outputs=loss_out)
        # else:
        #     return Model(inputs=[inputs], outputs=y_pred)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                           optimizer=self.config.model.optimizer)
        self.model.summary()

    def get_downsample_factor(self):
        return self.downsample_factor


if __name__ == '__main__':
    from utils.config import process_config
    config = process_config('../configs/config.json')
    model = CTCModel(config=config)
