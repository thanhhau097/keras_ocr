import cv2
import os
import random
import json
from sklearn.utils import shuffle
from utils.ocr_utils import labels_to_text, text_to_labels, get_input_token_index
import numpy as np
from data_loader import augmentions
from data_loader import policies as found_policies
from utils.ocr_utils import update_vocab


class OCRDataLoader(object):
    def __init__(self, config, phase='train'):
        self.config = config
        self.batch_size = self.config.trainer.batch_size
        self.max_text_len = self.config.hyperparameter.max_text_len
        self.max_height = self.config.image.max_height
        self.channels = self.config.image.channels
        self.phase = phase
        self.good_policies = found_policies.good_policies()

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

        # if phase == 'train':
        #     self.build_vocab()
        # elif phase == 'test': # no need val, because it is concurrent with train
        #     self.load_vocab()
        # print('Done initialization!')

    # def build_vocab(self):
    #     _, val_labels = self.get_image_paths_and_labels(self.get_data_path(self.config.data.val_json_path))
    #     letters = set()
    #     # add letters not in vocab files
    #     for label in (self.labels + val_labels):
    #         for char in label:
    #             if char not in letters:
    #                 letters.add(char)
    #
    #     letters = ''.join(list(letters))
    #     print('Number of characters:', len(letters))
    #     update_vocab(letters)
    #
    #     with open(self.config.data.vocab_path, 'w') as f:
    #         json.dump({'characters': letters}, f)
    #
    # def load_vocab(self):
    #     with open(self.config.data.vocab_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)
    #         update_vocab(data['characters'])


    def get_steps(self):
        return self.n // self.batch_size

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

                labels = np.zeros([self.batch_size, self.max_text_len], dtype=np.int)
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
                    if self.config.vocab_type == 'ctc':
                        label = text_to_labels(self.labels[j])
                    else:
                        label = text_to_labels(self.labels[j] + '\n')

                    labels[j - i * self.batch_size, :len(label)] = label[:self.config.hyperparameter.max_text_len]

                    if len(self.labels[j]) != 0:
                        if len(self.labels[j]) > self.config.hyperparameter.max_text_len:
                            label_length[j - i * self.batch_size] = self.config.hyperparameter.max_text_len
                        else:
                            label_length[j - i * self.batch_size] = len(self.labels[j])
                    else:
                        label_length[j - i * self.batch_size] = 1

                    if img.shape[1] > max_width:
                        max_width = img.shape[1]

                images = self.process_batch_images(images, max_width)
                images = np.transpose(images, [0, 2, 1, 3])

                # TODO ctc and attention
                if self.config.vocab_type == 'ctc':
                    input_length = np.ones((self.batch_size, 1)) * (max_width // self.config.downsample_factor - 2)

                    inputs = {
                        'the_input': images,  # (bs, w, h, 1)
                        'the_labels': labels,  # (bs, 8)
                        'input_length': input_length,  # (bs, 1)
                        'label_length': label_length,  # (bs, 1)
                    }
                    outputs = {'ctc': np.zeros([self.batch_size])}  # (bs, 1) -> 모든 원소 0
                    # print("images shape:", images.shape, "input_length:", list(input_length))

                    yield (inputs, outputs)
                else:
                    # for attention, we need decoder inputs: it is a constant value representing \t: start of sentence
                    # print(self.config.n_letters)
                    # print(self.config)
                    decoder_input_data = np.zeros([self.batch_size, 1, int(self.config.n_letters)])
                    decoder_input_data[:, 0, get_input_token_index()['\t']] = 1.

                    inputs = {
                        'the_input': images,
                        'decoder_input': decoder_input_data,
                    }
                    outputs = self.onehot_initialization(labels, self.config.n_letters)
                    yield (inputs, outputs)

    def onehot_initialization(self, a, ncols):
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self.all_idx(a, axis=2)] = 1
        return out

    def all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def process_batch_images(self, images, max_width):
        """Apply augmentations"""
        output = np.ones([self.batch_size, self.max_height, max_width, self.channels])
        for i, image in enumerate(images):
            final_img = image
            if self.config.augmentation:
                epoch_policy = self.good_policies[np.random.choice(
                    len(self.good_policies))]
                final_img = augmentions.apply_policy(
                    epoch_policy, final_img)
                # final_img = augmentions.random_flip(
                #     augmentions.zero_pad_and_crop(final_img, 4))
                # Apply cutout
                final_img = augmentions.cutout_numpy(final_img)

            shape = image.shape
            output[i, :shape[0], :shape[1], :] = final_img
        return output


if __name__ == '__main__':
    from utils.config import process_config
    config = process_config('../configs/config.json')
    config.downsample_factor = 4
    dataloader = OCRDataLoader(config, phase="train")

    data = next(dataloader.next_batch())[0]['the_input']
    from matplotlib import pyplot as plt
    for image in data:
        plt.imshow(image.transpose([1, 0, 2]))
        plt.show()
