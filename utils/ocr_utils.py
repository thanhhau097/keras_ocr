def label_to_text(labels, letters):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text, letters):      # text를 letters 배열에서의 인덱스 값으로 변환
    # TODO wrong character embedding
    # try:
    #     return list(map(lambda x: letters.index(x), text))
    # except:
    #     print(text)
    #     return [0] * 30
    output = []
    for x in text:
        try:
            output.append(letters.index(x))
        except:
            print(x)
            return [0] * 10

    return output