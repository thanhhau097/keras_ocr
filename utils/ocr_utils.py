from keras import backend as K

def label_to_text(labels, letters):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text, letters):      # text를 letters 배열에서의 인덱스 값으로 변환
    # TODO wrong character embedding
    # try:
    return list(map(lambda x: letters.index(x), text))
    # except:
    #     print(text)
    #     return [0] * 30
    # output = []
    # for x in text:
    #     try:
    #         output.append(letters.index(x))
    #     except:
    #         print(x)
    #         return [0] * 10
    #
    # return output


# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
