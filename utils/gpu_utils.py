from keras.layers import GRU, LSTM
from keras.layers import CuDNNGRU, CuDNNLSTM
import tensorflow as tf



# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
# the code automatically does that.
if tf.test.is_gpu_available():
    gru = CuDNNGRU
    lstm = CuDNNLSTM
else:
    gru = GRU
    lstm = LSTM
