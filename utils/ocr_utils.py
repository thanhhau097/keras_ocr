from keras import backend as K
import numpy as np
import itertools


LETTERS = ''


def update_vocab(letters):
    global LETTERS
    LETTERS = letters


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(LETTERS):  # CTC Blank
            ret.append("")
        else:
            try:
                ret.append(LETTERS[c])
            except:
                print(c)
                raise ValueError('Index out of range')
    return "".join(ret)
    # return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: LETTERS.index(x), text))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(pred_func, word_batch):
    out = pred_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)] # best path
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret
