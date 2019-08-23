import keras
import editdistance
import numpy as np
import itertools
from utils.ocr_utils import labels_to_text, get_reverse_target_char_index
from tqdm import tqdm
import textdistance


# TODO https://github.com/keras-team/keras/issues/9914
class AttentionCallback(keras.callbacks.Callback):
    def __init__(self, letters, steps, batch_size, validation_data,
                 filepath, save_weights_only=False):
        super(AttentionCallback, self).__init__()
        self.letters = letters
        self.text_img_gen = validation_data
        self.validation_steps = steps
        self.batch_size = batch_size
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.min_loss = np.inf

    def decode_batch_validation(self, data_batch):
        predicts = self.model.predict(data_batch[0])
        labels = data_batch[1]  # labels one-hot vector
        predicted_labels = []
        ground_truth_labels = []

        # we need to calculate the cross entropy loss here
        loss = []
        for i in range(len(labels)):
            predicted_label = ''
            predict = predicts[i]
            for element in predict:
                c = np.argmax(element)
                char = get_reverse_target_char_index()[c]
                if char == '\n':
                    break
                predicted_label += char

            str_label = ''
            label = labels[i]
            for element in label:
                c = np.argmax(element)
                char = get_reverse_target_char_index()[c]
                if char == '\n':
                    break
                str_label += char

            predicted_labels.append(predicted_label)
            ground_truth_labels.append(str_label)
            # TODO wrong here, we need to calculate for each step?
            item_loss = np.mean(self.categorical_crossentropy(labels[i], predicts[i]))
            # item_loss = self.cross_entropy(predicts[i], labels[i])
            loss.append(item_loss)

        # loss /= self.batch_size
        # print(predicted_labels)
        # print(str_labels)
        return predicted_labels, ground_truth_labels, loss

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
        mean_norm_ed = 0.0
        mean_ed = 0.0
        loss_batch = 0
        true_fields = 0
        acc_by_char = 0
        while num_left > 0:
            data_batch = next(self.text_img_gen)

            # we can use model.predict here, because the output of attention model is only y_pred, not the loss like ctc
            decoded_res, str_labels, loss = self.decode_batch_validation(data_batch)
            loss_batch = np.sum(loss)

            for j in range(num_left):
                # label_length = int(data_batch['label_length'][j])
                # source_str = labels_to_text(data_batch['the_labels'][j])[:label_length]
                source_str = str_labels[j]
                edit_dist = editdistance.eval(decoded_res[j], source_str)
                levenshtein_distance = textdistance.levenshtein.normalized_similarity(source_str, decoded_res[j])
                acc_by_char += levenshtein_distance
                mean_ed += float(edit_dist)
                if len(source_str) != 0:
                    mean_norm_ed += float(edit_dist) / len(source_str)
                else:
                    mean_norm_ed += float(edit_dist)
                if decoded_res[j] == source_str:
                    true_fields += 1

            num_left -= num
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        loss_batch = loss_batch / num
        acc_by_char = acc_by_char / num

        return mean_norm_ed, mean_ed, loss_batch, true_fields, acc_by_char

    def on_epoch_end(self, epoch, logs=None):
        """Calculate accuracy for final train batch and accuracy for validation set
        """
        print("Log:", logs)
        total_mean_norm_ed = 0
        total_mean_ed = 0
        total_loss = 0
        total_true_fields = 0
        total_acc_by_char = 0

        print("Evaluating Validation set ...")
        for _ in tqdm(range(self.validation_steps)):
            mean_norm_ed, mean_ed, loss_batch, true_fields, acc_by_char = self.show_edit_distance(self.batch_size)
            total_mean_norm_ed += mean_norm_ed
            total_mean_ed += mean_ed
            total_loss += loss_batch
            total_true_fields += true_fields
            total_acc_by_char += acc_by_char

        total_mean_norm_ed /= self.validation_steps
        total_mean_ed /= self.validation_steps
        total_loss /= self.validation_steps
        accuracy_by_field = total_true_fields / (self.validation_steps * self.batch_size)
        total_acc_by_char /= self.validation_steps

        print('\nMean edit distance:'
              '%.3f \nMean normalized edit distance: %0.3f'
              '\nLoss batch: %0.3f'
              '\nAccuracy by fields: %0.3f'
              '\nAccuracy by chars: %0.3f'
              % (total_mean_ed, total_mean_norm_ed, total_loss, accuracy_by_field, total_acc_by_char))

        if total_loss < self.min_loss:
            self.min_loss = total_loss
            print("Update new weights")
            if self.save_weights_only:
                self.model.save_weights(self.filepath, overwrite=True)
            else:
                self.model.save(self.filepath, overwrite=True)
