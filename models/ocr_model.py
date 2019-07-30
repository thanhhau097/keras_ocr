from utils.ocr_utils import ctc_lambda_func
from base.base_model import BaseModel
from keras.models import Model
from keras.layers import *
from models.encoders.simple_encoder import SimpleEncoder
from models.decoders.simple_decoder import SimpleDecoder


class OCRModel(BaseModel):
    def __init__(self, config):
        super(OCRModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        input_shape = (128, 64, 1)  # (128, 64, 1)

        # Make Networkw
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

        # ENCODER
        encoder = SimpleEncoder()
        inner = encoder(inputs)
        print("After Encoder:", inner)

        # CNN to RNN
        shape = inner.shape
        target_shape = (int(shape[1]), int(shape[2] * shape[3]))
        inner = Reshape(target_shape=target_shape, name='reshape')(inner)  # (None, 32, 2048)
        inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)
        print("After CNN to RNN:", inner)

        # DECODER
        decoder = SimpleDecoder()
        inner = decoder(inner)
        print("After Decoder:", inner)

        # transforms RNN output to character activations:
        num_classes = 100
        inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        max_text_len = 100
        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')  # (None ,8)
        input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
        label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])  # (None, 1)

        # if training:
        model =  Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        # else:
        #     return Model(inputs=[inputs], outputs=y_pred)



if __name__ == '__main__':
    model = OCRModel(config={})
