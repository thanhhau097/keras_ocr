import os
import cv2
import json
import tensorflow as tf

from models.ocr_model import OCRModel
from utils.config import process_config
from utils.ocr_utils import decode_batch
from utils.ocr_utils import update_vocab


DEFAULT_CONFIG_PATH = 'configs/config.json'


# global graph
graph = tf.get_default_graph()


class LionelOCR(object):
    def __init__(self,
                 weights_path,
                 config_path=DEFAULT_CONFIG_PATH):
        """OCR model with CNN encoder and LSTM-CTC decoder
        Parameters
        ----------
        weights_path : str or pathlib.Path
            path to model saved weights
        config_path : str or pathlib.Path
            path to config json file
        """
        self.weights = weights_path
        self.config = process_config(config_path)
        self._load_vocab()
        self.model = OCRModel(self.config)
        self.model.load(self.weights)

    def _load_vocab(self):
        with open(self.config.data.vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            update_vocab(data['characters'])

    def _preprocess(self):
        pass

    def _process_batch(self):
        pass

    def process(self):
        pass