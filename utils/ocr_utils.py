from keras import backend as K
import numpy as np
import itertools
import os
import json


LETTERS = ''
input_token_index = dict()
target_token_index = dict()


def build_vocab(config):
    vocab_type = config.vocab_type
    assert vocab_type in ['ctc', 'attention']
    _, train_labels = get_image_paths_and_labels(get_data_path(config, config.data.train_json_path))
    _, val_labels = get_image_paths_and_labels(get_data_path(config, config.data.val_json_path))
    letters = set()
    # add letters not in vocab files
    for label in (train_labels + val_labels):
        for char in label:
            if char not in letters:
                letters.add(char)

    if vocab_type == 'ctc':
        letters = ''.join(list(letters))
        print('Number of characters:', len(letters))
    else:  # 'attention'
        letters = list(letters)
        letters += ['\t', '\n'] # start and end token
    update_vocab(letters)
    # with open('data/' + config.data.vocab_path, 'w') as f:
    #     json.dump({'characters': letters}, f)

    return len(letters)


def load_vocab(config):
    with open(config.data.vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # update_vocab(data['characters'])


def get_data_path(config, path):
    return os.path.join(config.data.root, path)


def get_image_paths_and_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_paths = list(data.keys())
    labels = list(data.values())
    return image_paths, labels


def update_vocab(letters):
    global LETTERS, input_token_index, target_token_index
    LETTERS = list(letters)
    input_token_index = dict(
        [(char, i) for i, char in enumerate(LETTERS)]
    )

    target_token_index = dict(
        [(char, i) for i, char in enumerate(LETTERS)]
    )


def get_input_token_index():
    global input_token_index
    return input_token_index


def get_target_token_index():
    global target_token_index
    return target_token_index


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
