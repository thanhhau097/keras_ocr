from keras.layers import *
from keras.layers.merge import add, concatenate


class AttentionDecoder():
    def __init__(self, config):
        # used for attention: Bahdanau
        self.config = config
        latent_dim = 512
        num_decoder_tokens = self.config.n_letters
        self.max_decoder_seq_length = self.config.hyperparameter.max_text_len

        self.W1 = Dense(latent_dim)
        self.W2 = Dense(latent_dim)
        self.V = Dense(1)

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(1, num_decoder_tokens), name='decoder_input')
        self.decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    def __call__(self, input_tensor, *args, **kwargs):
        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(256, return_sequences=True,
                    kernel_initializer='he_normal', name='gru1')(input_tensor)
        gru_1b = GRU(256, return_sequences=True,
                     go_backwards=True, kernel_initializer='he_normal',
                     name='gru1_b')(input_tensor)
        gru1_merged = add([gru_1, gru_1b])
        gru_2, state_2 = GRU(256, return_sequences=True,
                         kernel_initializer='he_normal',
                         return_state=True, name='gru2')(gru1_merged)
        gru_2b, state_2b = GRU(256, return_sequences=True, go_backwards=True,
                          kernel_initializer='he_normal',
                          return_state=True, name='gru2_b')(gru1_merged)

        # transforms RNN output to character activations:
        encoder_outputs = concatenate([gru_2, gru_2b])
        state = concatenate([state_2, state_2b])

        # Apply Attention
        all_outputs = []
        inputs = self.decoder_inputs
        for _ in range(self.max_decoder_seq_length):
            _, state = self.decoder_gru(inputs, initial_state=state)
            attention_v = self.score_module(encoder_outputs, state)
            outputs = self.decoder_dense(attention_v)

            inputs = Lambda(lambda x: K.expand_dims(x, axis=1))(outputs)
            all_outputs.append(inputs)
        decoder_outputs = Concatenate(axis=1)(all_outputs)

        print("DECODER_OUTPUTS: ", decoder_outputs)
        return decoder_outputs, self.decoder_inputs

    def score_module(self, encoder_outputs, state):  # Bahdanau
        hidden_state_with_time_axis = Lambda(lambda x: K.expand_dims(x, axis=1))(state)  # K.expand_dims(state, 1)
        output_hidden = self.W2(hidden_state_with_time_axis)
        tanh_lambda = Lambda(lambda x: K.tanh(x))
        score = Add()([tanh_lambda(self.W1(encoder_outputs)), output_hidden])
        score = self.V(score)
        attention_weights = Softmax(axis=1)(score)
        context_vector = Multiply()([attention_weights, encoder_outputs])
        sum_lambda = Lambda(lambda x: K.sum(x, axis=1))
        context_vector = sum_lambda(context_vector)
        attention_vector = Concatenate(axis=-1)([context_vector, state])

        return attention_vector
