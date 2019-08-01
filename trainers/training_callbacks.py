import keras
import editdistance
import numpy as np
import itertools
from utils.ocr_utils import label_to_text
from utils.ocr_utils import decode_batch_validation


# https://github.com/keras-team/keras/issues/10472

class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, test_func, letters, steps, batch_size, validation_data):
        super(TrainingCallback, self).__init__()
        self.test_func = test_func
        self.letters = letters
        self.text_img_gen = validation_data
        self.validation_steps = steps
        self.batch_size = batch_size

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        loss_batch = 0
        while num_left > 0:
            # word_batch = next(self.text_img_gen)[0]
            data_batch = next(self.text_img_gen)[0]
            # word_batch = data_batch[0]
            # num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res, loss = decode_batch_validation(self.test_func,
                                                        data_batch,
                                                        self.letters)
            loss_batch = np.sum(loss)

            for j in range(num_left):
                label_length = int(data_batch['label_length'][j])
                source_str = label_to_text(data_batch['the_labels'][j], self.letters)[:label_length]
                print('source_str:', source_str)
                print('predicted_str', decoded_res[j])
                edit_dist = editdistance.eval(decoded_res[j], source_str)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(source_str)
            num_left -= num
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        loss_batch = loss_batch / num

        print('\nOut of %d samples:  Mean edit distance:'
              '%.3f Mean normalized edit distance: %0.3f'
              'Loss batch: %0.3f'
              % (num, mean_ed, mean_norm_ed, loss_batch))

    def on_epoch_end(self, epoch, logs=None):
        """Calculate accuracy for final train batch and accuracy for validation set
        """
        for step in range(self.validation_steps):
            self.show_edit_distance(self.batch_size)


