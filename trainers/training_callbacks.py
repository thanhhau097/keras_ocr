import keras
import editdistance
import numpy as np
import itertools
from utils.ocr_utils import label_to_text


# https://github.com/keras-team/keras/issues/10472

class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, test_func, letters, steps, batch_size, validation_data):
        super(TrainingCallback, self).__init__()
        self.test_func = test_func
        self.letters = letters
        self.text_img_gen = validation_data
        self.validation_steps = steps
        self.batch_size = batch_size

    def decode_batch(self, test_func, word_batch):
        out = test_func([word_batch])[0]
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = label_to_text(out_best, self.letters)
            ret.append(outstr)
        return ret

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = self.decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                label_length = int(word_batch['label_length'][j])
                source_str = label_to_text(word_batch['the_labels'][j], self.letters)[:label_length]
                print('source_str:', source_str)
                print('predicted_str', decoded_res[j])
                edit_dist = editdistance.eval(decoded_res[j], source_str)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(source_str)
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance:'
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs=None):
        """Calculate accuracy for final train batch and accuracy for validation set
        """
        for step in range(self.validation_steps):
            self.show_edit_distance(self.batch_size)


