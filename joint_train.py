#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
# from nnet.py_factory import NetworkFactory
from nnet.joint_py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
import models.py_utils.misc as utils
from db.utils.evaluator import Evaluator

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--train_mode", dest="train_mode", default='sequential', type=str)
    parser.add_argument("--test_mode", dest="test_mode", default='PvTv', type=str)
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def prefetch_data(db, queue, sample_data):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]
        # data["ys"] = [y for y in data["ys"]]
        pinned_data_queue.put(data)
        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, testing_db, start_iter=0, freeze=False, train_mode=None, test_mode=None):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size) # 5
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size) # 5
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # print(type(sample_data)) # function

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(training_dbs[0], flag=True, freeze=freeze, train_mode=train_mode, test_mode=test_mode)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    metric_logger.add_meter('pv_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('tv_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    os.makedirs(os.path.join(system_configs.result_dir, 'tb_train'), exist_ok=True)
    os.makedirs(os.path.join(system_configs.result_dir, 'tb_eval'), exist_ok=True)
    train_writer = SummaryWriter(log_dir=os.path.join(system_configs.result_dir, 'tb_train'))
    eval_writer  = SummaryWriter(log_dir=os.path.join(system_configs.result_dir, 'tb_eval'))

    with stdout_to_tqdm() as save_stdout:
        for iteration in metric_logger.log_every(
                tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=60), print_freq=10, header=header):

            training = pinned_training_queue.get(block=True)
            viz_split = 'train'
            # save = True if (display and iteration % display == 0) else False
            (set_loss, pv_loss_dict, tv_loss_dict) \
                = nnet.train(iteration, False, viz_split, **training)
            (pv_loss_dict_reduced, pv_loss_dict_reduced_unscaled, pv_loss_dict_reduced_scaled, pv_loss_value) = pv_loss_dict
            (tv_loss_dict_reduced, tv_loss_dict_reduced_unscaled, tv_loss_dict_reduced_scaled, tv_loss_value) = tv_loss_dict
            # metric_logger.update(pv_loss=pv_loss_value, tv_loss=tv_loss_value,
            #                      **pv_loss_dict_reduced_scaled, **pv_loss_dict_reduced_unscaled,
            #                      **tv_loss_dict_reduced_scaled, **tv_loss_dict_reduced_unscaled)
            metric_logger.update(pv_loss=pv_loss_value, tv_loss=tv_loss_value)
            metric_logger.update(pv_class_error=pv_loss_dict_reduced['class_error'])
            metric_logger.update(tv_class_error=tv_loss_dict_reduced['class_error'])
            # metric_logger.update(lr=nnet.scheduler.get_lr()[0])
            metric_logger.update(lr=learning_rate)

            if iteration % 10 == 0:
                train_writer.add_scalar("Train/set_loss", set_loss, iteration)
                # for k, v in pv_loss_dict_reduced_unscaled.items():
                #     train_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in pv_loss_dict_reduced_scaled.items():
                #     train_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in tv_loss_dict_reduced_unscaled.items():
                #     train_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in tv_loss_dict_reduced_scaled.items():
                #     train_writer.add_scalar("Train/{}".format(k), v, iteration)

            del set_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                viz_split = 'val'
                validation = pinned_validation_queue.get(block=True)
                (val_set_loss, val_pv_loss_dict, val_tv_loss_dict) \
                    = nnet.validate(iteration, False, viz_split, **validation)
                (pv_loss_dict_reduced, pv_loss_dict_reduced_unscaled, pv_loss_dict_reduced_scaled, pv_loss_value) = val_pv_loss_dict
                (tv_loss_dict_reduced, tv_loss_dict_reduced_unscaled, tv_loss_dict_reduced_scaled, tv_loss_value) = val_tv_loss_dict
                # metric_logger.update(pv_loss=pv_loss_value, tv_loss=tv_loss_value,
                #                      **pv_loss_dict_reduced_scaled, **pv_loss_dict_reduced_unscaled,
                #                      **tv_loss_dict_reduced_scaled, **tv_loss_dict_reduced_unscaled)
                metric_logger.update(pv_loss=pv_loss_value, tv_loss=tv_loss_value)
                metric_logger.update(pv_class_error=pv_loss_dict_reduced['class_error'])
                metric_logger.update(tv_class_error=tv_loss_dict_reduced['class_error'])
                # metric_logger.update(lr=nnet.scheduler.get_lr()[0])
                metric_logger.update(lr=learning_rate)
                eval_writer.add_scalar("Eval/set_loss", val_set_loss, iteration)
                # for k, v in pv_loss_dict_reduced_unscaled.items():
                #     eval_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in pv_loss_dict_reduced_scaled.items():
                #     eval_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in tv_loss_dict_reduced_unscaled.items():
                #     eval_writer.add_scalar("Train/{}".format(k), v, iteration)
                # for k, v in tv_loss_dict_reduced_scaled.items():
                #     eval_writer.add_scalar("Train/{}".format(k), v, iteration)
                nnet.train_mode()
            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            # if iteration >= stepsize:
            #     if iteration % (training_size // batch_size) == 0:
            #         nnet.save_params(iteration)
            #         nnet.eval_mode()
            #         test_file = "test.fast_{}".format(testing_db._data)
            #         print('test_file: {}'.format(test_file))
            #         testing = importlib.import_module(test_file).testing
            #         result_dir = os.path.join(system_configs.result_dir, str(iteration), 'Evaluation')
            #         make_dirs([result_dir])
            #         print('[LOG][Static evaluating on {}...]'.format(result_dir))
            #         evaluator = Evaluator(testing_db, result_dir)
            #         test_stats = testing(testing_db, nnet, result_dir, debug=False, evaluator=evaluator)
            #         eval_writer.add_scalar("Stats/F1", test_stats['F-score'], iteration)
            #         eval_writer.add_scalar("Stats/AP", test_stats["AP"], iteration)
            #         eval_writer.add_scalar("Stats/E_height", test_stats["EH"], iteration)
            #         eval_writer.add_scalar("Stats/E_pitch", test_stats["EP"], iteration)
            #         metric_logger.synchronize_between_processes()
            #         print("Averaged stats:", metric_logger)
            #         train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                      **{f'test_{k}': v for k, v in test_stats.items()},
            #                      'iteration': iteration}
            #         if system_configs.result_dir and utils.is_main_process():
            #             with (Path(system_configs.result_dir) / "log.txt").open("a") as f:
            #                 f.write(json.dumps(log_stats) + "\n")
            #             f.close()
            #         nnet.train_mode()
            # else:
            #     if iteration % snapshot == 0:
            #         nnet.save_params(iteration)
            #         # nnet.eval_mode()
            #         # test_file = "test.fast_{}".format(testing_db._data)
            #         # print('test_file: {}'.format(test_file))
            #         # testing = importlib.import_module(test_file).testing
            #         # result_dir = os.path.join(system_configs.result_dir, str(iteration), 'Evaluation')
            #         # make_dirs([result_dir])
            #         # print('[LOG][Static evaluating on {}...]'.format(result_dir))
            #         # evaluator = Evaluator(testing_db, result_dir)
            #         # test_stats = testing(testing_db, nnet, result_dir, debug=False, evaluator=evaluator)
            #         # eval_writer.add_scalar("Stats/F1", test_stats['F-score'], iteration)
            #         # eval_writer.add_scalar("Stats/AP", test_stats["AP"], iteration)
            #         # eval_writer.add_scalar("Stats/E_height", test_stats["EH"], iteration)
            #         # eval_writer.add_scalar("Stats/E_pitch", test_stats["EP"], iteration)
            #         # metric_logger.synchronize_between_processes()
            #         # print("Averaged stats:", metric_logger)
            #         # train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            #         # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #         #              **{f'test_{k}': v for k, v in test_stats.items()},
            #         #              'iteration': iteration}
            #         # if system_configs.result_dir and utils.is_main_process():
            #         #     with (Path(system_configs.result_dir) / "log.txt").open("a") as f:
            #         #         f.write(json.dumps(log_stats) + "\n")
            #         #     f.close()
            #         # nnet.train_mode()


    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = args.cfg_file  # CornerNet
    system_configs.update_config(configs["system"])
    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    test_split  = system_configs.test_split
    dataset = system_configs.dataset
    print("loading all datasets {}...".format(dataset))
    threads = args.threads  # 4 every 4 epoch shuffle the indices
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)
    testing_db    = datasets[dataset](configs["db"], test_split, is_eval=True, is_predcam=False)
    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(training_dbs[0].configs)
    print("len of training db: {}".format(len(training_dbs[0].db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))
    print("freeze the pretrained network: {}".format(args.freeze))
    print("training mode: {}".format(args.train_mode))
    print("testing mode: {}".format(args.test_mode))
    train(training_dbs, validation_db, testing_db, args.start_iter, args.freeze, args.train_mode, args.test_mode) # 0
