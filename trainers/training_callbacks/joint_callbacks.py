import keras
import editdistance
import numpy as np
from utils.ocr_utils import labels_to_text, text_to_labels, get_reverse_target_char_index
from tqdm import tqdm
import itertools


# TODO https://github.com/keras-team/keras/issues/9914
class JointCallback(keras.callbacks.Callback):
    """Callback class for Joint CTC-Attention Model"""
    def __init__(self, test_func, letters, steps, batch_size, validation_data,
                 filepath, save_weights_only=False):
        super(JointCallback, self).__init__()
        self.test_func = test_func
        self.letters = letters
        self.text_img_gen = validation_data
        self.validation_steps = steps
        self.batch_size = batch_size
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.min_loss = np.inf

    # TODO wrong attention loss
    def decode_batch_validation(self, data_batch):
        # we should use test function here, for both y_pred_ctc, and y_pred_attention
        y_pred_ctc, loss_ctc, y_pred_attention = self.test_func(list(data_batch[0].values()))

        # ------------- CTC -----------
        result_ctc = []
        for j in range(y_pred_ctc.shape[0]):
            out_best = list(np.argmax(y_pred_ctc[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            result_ctc.append(outstr)
        loss_ctc = np.reshape(loss_ctc, [-1])

        # ------------ Attention --------------
        # don't need to re-calculate ground_truth_labels (like attention callback) because we
        # already have had it
        labels = data_batch[1]['attention']

        result_attention = []
        loss_attention = []

        for i in range(len(labels)):
            predicted_label = ''
            predict = y_pred_attention[i]
            for element in predict:
                c = np.argmax(element)
                char = get_reverse_target_char_index()[c]
                if char == '\n':
                    break
                predicted_label += char

            result_attention.append(predicted_label)
            item_loss = 0
             # TODO wrong here, we need to calculate for each step?
            # for j in range(len(labels[i])):
            #     item_loss += self.cross_entropy(y_pred_attention[i][j], labels[i][j])
            # item_loss = self.cross_entropy(y_pred_attention[i], labels[i])
            item_loss = np.mean(self.categorical_crossentropy(labels[i], y_pred_attention[i]))
            loss_attention.append(item_loss)

        return result_ctc, loss_ctc, result_attention, loss_attention

    # def cross_entropy(self, predictions, targets, epsilon=1e-12):
    #     """
    #     Computes cross entropy between targets (encoded as one-hot vectors)
    #     and predictions.
    #     Input: predictions (N, k) ndarray
    #            targets (N, k) ndarray
    #     Returns: scalar
    #     """
    #     predictions = np.clip(predictions, epsilon, 1. - epsilon)
    #     ce = - np.mean(np.log(predictions) * targets)
    #     return ce

    def categorical_crossentropy(self, target, output):
        output /= output.sum(axis=-1, keepdims=True)
        output = np.clip(output, 1e-7, 1 - 1e-7)
        return np.sum(target * -np.log(output), axis=-1, keepdims=False)

    def show_edit_distance(self, num):
        num_left = num
        ctc_mean_norm_ed = 0.0
        ctc_mean_ed = 0.0
        ctc_loss_batch = 0
        ctc_true_fields = 0
        attention_mean_norm_ed = 0.0
        attention_mean_ed = 0.0
        attention_loss_batch = 0
        attention_true_fields = 0
        while num_left > 0:
            data_batch = next(self.text_img_gen)
            result_ctc, loss_ctc, result_attention, loss_attention = self.decode_batch_validation(data_batch)

            ctc_loss_batch = np.sum(loss_ctc)
            attention_loss_batch = np.sum(loss_attention)

            for j in range(num_left):

                label_length = int(data_batch[0]['label_length'][j])
                source_str = labels_to_text(data_batch[0]['the_labels'][j])[:label_length]

                # -------- CTC -----------
                edit_dist = editdistance.eval(result_ctc[j], source_str)
                ctc_mean_ed += float(edit_dist)
                ctc_mean_norm_ed += float(edit_dist) / len(source_str)
                if result_ctc[j] == source_str:
                    ctc_true_fields += 1

                # ------- Attention ---------
                edit_dist = editdistance.eval(result_attention[j], source_str)
                attention_mean_ed += float(edit_dist)
                if len(source_str) != 0:
                    attention_mean_norm_ed += float(edit_dist) / len(source_str)
                else:
                    attention_mean_norm_ed += float(edit_dist)
                if result_attention[j] == source_str:
                    attention_true_fields += 1

            num_left -= num

        ctc_mean_norm_ed = ctc_mean_norm_ed / num
        ctc_mean_ed = ctc_mean_ed / num
        ctc_loss_batch = ctc_loss_batch / num

        attention_mean_norm_ed = attention_mean_norm_ed / num
        attention_mean_ed = attention_mean_ed / num
        attention_loss_batch = attention_loss_batch / num

        return ctc_mean_norm_ed, ctc_mean_ed, ctc_loss_batch, ctc_true_fields, \
            attention_mean_norm_ed, attention_mean_ed, attention_loss_batch, attention_true_fields

    def on_epoch_end(self, epoch, logs=None):
        ctc_metrics = np.zeros(4)
        attention_metrics = np.zeros(4)

        print("Evaluating Validation set ...")
        for _ in tqdm(range(self.validation_steps)):
            metrics = self.show_edit_distance(self.batch_size)
            ctc_metrics += np.array(list(metrics[0:4]))
            attention_metrics += np.array(list(metrics[4:]))

        print('\n---- CTC ------'
              '\nMean edit distance:'
              '%.3f \nMean normalized edit distance: %0.3f'
              '\nLoss batch: %0.3f'
              '\nAccuracy by fields: %0.3f'
              % (ctc_metrics[0], ctc_metrics[1], ctc_metrics[2], ctc_metrics[3]))

        print('\n---- Attention ------'
              '\nMean edit distance:'
              '%.3f \nMean normalized edit distance: %0.3f'
              '\nLoss batch: %0.3f'
              '\nAccuracy by fields: %0.3f'
              % (attention_metrics[0], attention_metrics[1], attention_metrics[2], attention_metrics[3]))

        total_loss = ctc_metrics[2] + attention_metrics[2]
        if total_loss < self.min_loss:
            self.min_loss = total_loss
            print("Update new weights")
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)



