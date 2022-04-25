#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import pprint
import argparse
import importlib
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")

from config import system_configs
# from nnet.py_factory import NetworkFactory
from nnet.joint_py_factory import NetworkFactory
from db.datasets import datasets
from db.utils.evaluator import Evaluator

torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test CLGo")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=500000, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="testing", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--video_root", dest="video_root",
                        default=None, type=str)
    parser.add_argument("--modality", dest="modality",
                        default="eval", type=str)
    parser.add_argument("--image_root", dest="image_root",
                        default=None, type=str)
    parser.add_argument("--predcam", action="store_true")
    parser.add_argument("--test_mode", dest="test_mode",
                        default="PvTv", type=str)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(db, split, testiter,
         debug=False, suffix=None, video_root=None, modality=None, image_root=None, test_mode=None):
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)
    make_dirs([result_dir])
    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    print("building neural network...")
    # db.batch_size = 1
    nnet = NetworkFactory(db, test_mode=test_mode)

    print("loading parameters...")
    nnet.load_params(test_iter)
    nnet.cuda()
    nnet.eval_mode()

    evaluator = Evaluator(db, result_dir)

    if modality == 'eval':
        print('static evaluating using [groundtruth] box...')
        print(db._data)
        test_file = "test.fast_{}".format(db._data)
        testing = importlib.import_module(test_file).testing
        testing(db, nnet, result_dir, debug=debug, evaluator=evaluator)

    elif modality == 'video':
        # raise NotImplementedError
        if video_root == None:
            raise ValueError('--video_root is not defined!')
        print("processing [video]...")
        test_file = "test.video".format(db._data)
        video_testing = importlib.import_module(test_file).testing
        video_testing(db, nnet, video_root, debug=debug, evaluator=None)

    elif modality == 'image':
        # raise NotImplementedError
        if image_root == None:
            raise ValueError('--image_root is not defined!')
        print("processing [images]...")
        test_file = "test.image".format(db._data)
        image_testing = importlib.import_module(test_file).testing
        image_testing(db, nnet, image_root, debug=debug, evaluator=None)

    else:
        raise ValueError('--modality must be one of video/images_and_evaluation/images_and_demo, but now: {}'
                         .format(modality))

if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    test_split  = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))  # test

    is_eval = not args.debug
    is_predcam = args.predcam
    print('is_eval: {}'.format(is_eval))
    print('is_predcam: {}'.format(is_predcam))
    testing_db = datasets[dataset](configs["db"], split, is_eval=is_eval, is_predcam=is_predcam)
    # print('testing_db.batch_size: {}'.format(testing_db.batch_size))

    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(testing_db.configs)
    print('test_mode: {}'.format(args.test_mode))
    test(testing_db,
         args.split,
         args.testiter,
         args.debug,
         args.suffix,
         args.video_root,
         args.modality,
         args.image_root,
         args.test_mode)
