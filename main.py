#!/usr/bin/python

from __future__ import print_function
import argparse, os, json, traceback, sys

import sys 
import os
import time 
import argparse
import random
import datetime
import subprocess

from utils.utils import (
    unzip, write_log, JsonHandler
)


class SagemakerInference(object):
    """ Configurations for setup env and training models 

        Arguments:
            config_path (str): path of configuration json file

    """
    def __init__(self, config_path):
        # read the config file to a config dict 
        self.json_tools = JsonHandler()
        self.configs = self.json_tools.read_json_file(config_path)

        # setup dependencies  
        if self.configs['setup'] == True:
            if os.path.isfile(self.configs['requirement_file']):
                print("Installing requirements...")
                subprocess.run("pip install -r {0}".format(\
                    self.configs['requirement_file']), shell=True)

    def unzip_data(self):
        if self.configs["data_zip_file"]:
            zip_file = self.configs["data_zip_file"]
            folder = os.path.dirname(zip_file)
            print("Doing unzip file {0}:".format(zip_file))
            unzip(zip_file, folder)

    def process(self):
        # import torch 
        # import numpy as np
        # import torch.backends.cudnn as cudnn

        # print('torch.cuda.is_available(): ', torch.cuda.is_available())

        # # Seed and GPU setting
        # random.seed(self.configs["manual_seed"])
        # np.random.seed(self.configs["manual_seed"])
        # torch.manual_seed(self.configs["manual_seed"])

        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.configs["manual_seed"])
        #     cudnn.benchmark = True
        #     cudnn.deterministic = True

        # unzip data
        self.unzip_data()
        
        os.system('nvidia-smi')
        # subprocess.run('sh run.sh', shell=True)
        # print("CONVERT TXT TO JSON ...")
        # subprocess.run('python convert_txt_to_json.py', shell=True)
        print("RUN TRAIN.PY FILE")
        subprocess.run('python train.py', shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for training OpenNMT-py')
    parser.add_argument('--config')
    args = parser.parse_known_args()[0]
    config_path = args.config

    inference = SagemakerInference(config_path)

    try:
        inference.process()
        sys.exit(0)
    except Exception as e:
        # Write out an error file. This will be returned as 
        # the failure reason in the describe training job result.
        trc = traceback.format_exc()
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        sys.exit(255)
