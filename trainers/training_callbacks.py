import keras
import editdistance
import numpy as np
import itertools
from utils.ocr_utils import label_to_text


# https://github.com/keras-team/keras/issues/10472

class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, test_func, letters, steps, batch_size, validation_data,
                 filepath, save_weights_only=False):
        super(TrainingCallback, self).__init__()
        self.test_func = test_func
        self.letters = letters
        self.text_img_gen = validation_data
        self.validation_steps = steps
        self.batch_size = batch_size
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.min_loss = np.inf

    def decode_batch_validation(self, test_func, data_batch, letters):
        output = test_func(list(data_batch.values()))
        out, loss = output
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = label_to_text(out_best, letters)
            ret.append(outstr)
        loss = np.reshape(loss, [-1])
        print(loss)
        return ret, loss

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        loss_batch = 0
        while num_left > 0:
            data_batch = next(self.text_img_gen)[0]
            decoded_res, loss = self.decode_batch_validation(self.test_func,
                                                             data_batch,
                                                             self.letters)
            loss_batch = np.sum(loss)

            for j in range(num_left):
                label_length = int(data_batch['label_length'][j])
                source_str = label_to_text(data_batch['the_labels'][j], self.letters)[:label_length]
                edit_dist = editdistance.eval(decoded_res[j], source_str)
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(source_str)
            num_left -= num
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        loss_batch = loss_batch / num

        return mean_norm_ed, mean_ed, loss_batch

    def on_epoch_end(self, epoch, logs=None):
        """Calculate accuracy for final train batch and accuracy for validation set
        """
        print("Log:", logs)
        total_mean_norm_ed = 0
        total_mean_ed = 0
        total_loss = 0

        for step in range(self.validation_steps):
            mean_norm_ed, mean_ed, loss_batch = self.show_edit_distance(self.batch_size)
            total_mean_norm_ed += mean_norm_ed
            total_mean_ed += mean_ed
            total_loss += loss_batch

        total_mean_norm_ed /= self.validation_steps
        total_mean_ed /= self.validation_steps
        total_loss /= self.validation_steps

        print('\nMean edit distance:'
              '%.3f \tMean normalized edit distance: %0.3f'
              '\tLoss batch: %0.3f'
              % (total_mean_ed, total_mean_norm_ed, total_loss))

        if total_loss < self.min_loss:
            self.min_loss = total_loss
            print("Update new weights")
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
