import cv2
import os
import random
import json
from sklearn.utils import shuffle
from utils.ocr_utils import label_to_text, text_to_labels
import numpy as np


class OCRDataLoader(object):
    def __init__(self, config, phase='train'):
        self.config = config
        self.batch_size = self.config.trainer.batch_size
        self.max_text_len = self.config.hyperparameter.max_text_len
        self.max_height = self.config.image.max_height
        self.channels = self.config.image.channels

        if phase == 'train':
            data_json_path = self.get_data_path(self.config.data.train_json_path)
        elif phase == 'val':
            data_json_path = self.get_data_path(self.config.data.val_json_path)
        elif phase == 'test':
            data_json_path = self.get_data_path(self.config.data.test_json_path)
        else:
            raise(ValueError("Phase must be in {'train', 'val', 'test'}"))

        self.image_paths, self.labels = self.get_image_paths_and_labels(data_json_path)
        self.n = len(self.image_paths)

        self.letters = self.read_letters()
        print('Done initialization!')
        print('Number of characters:', len(self.letters))

    def read_letters(self):
        with open(self.get_data_path(self.config.data.vocab_path), 'r') as f:
            data = json.load(f)

        letters = set(data.values())
        # add letters not in vocab files
        for label in self.labels:
            for char in label:
                if char not in letters:
                    letters.add(char)
        return list(letters)

    def get_data_path(self, path):
        return os.path.join(self.config.data.root, path)

    def get_image_paths_and_labels(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_paths = list(data.keys())
        labels = list(data.values())
        return image_paths, labels

    def process_image(self, image):
        """Converts to self.channels, self.max_height
        # convert channels
        # resize max_height = 64
        """
        shape = image.shape
        if shape[0] > 64 or shape[0] < 32: # height
            image = cv2.resize(image, (int(64/shape[0] * shape[1]), 64))
        return image / 255.0

    def next_sample(self):
        pass

    def next_batch(self):
        while True:
            self.image_paths, self.labels = shuffle(self.image_paths, self.labels)
            for i in range(self.n // self.batch_size):
                max_width = 0
                images = []
                labels = np.ones([self.batch_size, self.max_text_len], dtype=np.int)
                label_length = np.zeros((self.batch_size, 1))

                for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                    path = self.get_data_path(self.image_paths[j])
                    if self.channels == 1:
                        img = cv2.imread(path, 0)
                    elif self.channels == 3:
                        img = cv2.imread(path)
                    else:
                        raise ValueError("Number of channels must be 1 or 3")
                    img = self.process_image(image=img)
                    images.append(img)
                    # labels must be tensor `(samples, max_string_length)` containing the truth labels.
                    label = text_to_labels(self.labels[j], letters=self.letters)
                    labels[j - i * self.batch_size, :len(label)] = label
                    label_length[j - i * self.batch_size] = len(self.labels[j])

                    if img.shape[1] > max_width:
                        max_width = img.shape[1]
                # if max_width < 1024:
                #     max_width = 1024

                images = self.process_batch_images(images, max_width)
                images = np.transpose(images, [0, 2, 1, 3])
                input_length = np.ones((self.batch_size, 1)) * (max_width // self.config.downsample_factor - 2)
                inputs = {
                    'the_input': images,  # (bs, w, h, 1)
                    'the_labels': labels,  # (bs, 8)
                    'input_length': input_length,  # (bs, 1)
                    'label_length': label_length  # (bs, 1)
                }
                outputs = {'ctc': np.zeros([self.batch_size])}  # (bs, 1) -> 모든 원소 0
                # print("images shape:", images.shape, "input_length:", list(input_length))
                yield (inputs, outputs)

    def process_batch_images(self, images, max_width):
        output = np.ones([self.batch_size, self.max_height, max_width, self.channels])
        for i, image in enumerate(images):
            shape = image.shape
            output[i, :shape[0], :shape[1], :] = image
        return output


# if __name__ == '__main__':
#     from utils.config import process_config
#     config = process_config('../configs/config.json')
#     dataloader = OCRDataLoader(config, phase="train")
#
#     print(next(dataloader.next_batch())[0]['the_labels'])