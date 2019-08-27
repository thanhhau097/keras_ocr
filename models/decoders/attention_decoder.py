from keras.layers import *
from keras.layers.merge import add, concatenate, dot
from utils.gpu_utils import gru


class AttentionDecoder(object):
    def __init__(self, config):
        # used for attention: luong_attention
        self.config = config

        latent_dim = 512
        num_decoder_tokens = self.config.n_letters
        self.max_decoder_seq_length = self.config.hyperparameter.max_text_len

        self.W1 = Dense(latent_dim)
        self.W2 = Dense(latent_dim)
        self.V = Dense(1)

        # for luong general attention
        self.Wa = Dense(latent_dim)

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(1, num_decoder_tokens), name='decoder_input')
        self.decoder_gru = gru(latent_dim, return_sequences=True, return_state=True, name='decoder_gru')
        self.decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    def __call__(self, encoder_outputs, state, *args, **kwargs):
        # Apply Attention
        all_outputs = []
        inputs = self.decoder_inputs
        for _ in range(self.max_decoder_seq_length):
            _, state = self.decoder_gru(inputs, initial_state=state)
            attention_v = self.luong_general_attention(encoder_outputs, state)
            outputs = self.decoder_dense(attention_v)

            inputs = Lambda(lambda x: K.expand_dims(x, axis=1))(outputs)
            all_outputs.append(inputs)
        decoder_outputs = Concatenate(axis=1, name='attention')(all_outputs)

        print("DECODER_OUTPUTS: ", decoder_outputs)
        return decoder_outputs, self.decoder_inputs

    # TODO wrong: build from the previous hidden states
    # def bahdanau_score_module(self, encoder_outputs, state):  # Bahdanau
    #     """
    #     build from the previous hidden state ht−1 → at → ct → ht
    #
    #     :param encoder_outputs:
    #     :param state:
    #     :return:
    #     """
    #     hidden_state_with_time_axis = Lambda(lambda x: K.expand_dims(x, axis=1))(state)  # K.expand_dims(state, 1)
    #     tanh_lambda = Lambda(lambda x: K.tanh(x))
    #     score = Add()([tanh_lambda(self.W1(encoder_outputs)), self.W2(hidden_state_with_time_axis)])
    #     score = self.V(score)
    #     attention_weights = Softmax(axis=1)(score)
    #     context_vector = Multiply()([attention_weights, encoder_outputs])
    #     sum_lambda = Lambda(lambda x: K.sum(x, axis=1))
    #     context_vector = sum_lambda(context_vector)
    #     attention_vector = Concatenate(axis=-1)([context_vector, state])
    #
    #     return attention_vector

    # TODO https://arxiv.org/pdf/1508.04025.pdf
    def luong_dot_score_module(self, encoder_outputs, state):
        """
         from ht → at → ct → h˜t then make a prediction
        :param encoder_outputs:
        :param state:
        :return:
        """
        # print('state:', state)
        # print('encoder_outputs:', encoder_outputs)
        attention = dot([state, encoder_outputs], axes=[1, 2])
        attention = Activation('softmax')(attention)
        # print('attention', attention)

        context = dot([attention, encoder_outputs], axes=[1, 1])
        # print('context', context)

        decoder_combined_context = concatenate([context, state])
        # print('decoder_combined_context', decoder_combined_context)

        return decoder_combined_context

    def luong_general_attention(self, encoder_outputs, state):
        # print('state:', state)
        # print('encoder_outputs:', encoder_outputs)
        # difference here
        Wa_encoder_outputs = self.Wa(encoder_outputs)
        attention = dot([state, Wa_encoder_outputs], axes=[1, 2])
        attention = Activation('softmax')(attention)
        # print('attention', attention)

        context = dot([attention, encoder_outputs], axes=[1, 1])
        # print('context', context)

        decoder_combined_context = concatenate([context, state])
        # print('decoder_combined_context', decoder_combined_context)

        return decoder_combined_context
