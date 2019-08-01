from keras import backend as K
import numpy as np
import itertools


LETTERS = None

def label_to_text(labels, letters=None):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text, letters=None):      # text를 letters 배열에서의 인덱스 값으로 변환
    # TODO wrong character embedding
    # try:
    global LETTERS
    if LETTERS is None:
        LETTERS = letters
    return list(map(lambda x: letters.index(x), text))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = label_to_text(out_best, LETTERS)
        ret.append(outstr)
    return ret