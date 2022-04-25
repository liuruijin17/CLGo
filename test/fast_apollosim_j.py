import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from sample.vis import *
from models.py_utils.box_ops import *
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PLUM = (255, 187, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)
CORAL = (86, 114, 255)
CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)
GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [CORAL, GREEN, DARK_GREEN, PLUM, CHOCOLATE, PEACHPUFF, STATEGRAY]

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        results = torch.cat([scores.unsqueeze(-1), labels.unsqueeze(-1).float(), out_bbox], dim=-1)

        return results

def kp_detection(db, nnet, result_dir, debug=False, evaluator=None, attn_map=False):
    if db.split != "train":
        db_inds = db.db_inds if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds
    num_images = db_inds.size
    # num_images = 1
    input_size  = db.configs["input_size"]

    postprocessors = {'bbox': PostProcess()}
    test_batch_size = 16
    test_epoch = num_images // test_batch_size + 1
    for epid in tqdm(range(0, test_epoch), ncols=60, desc="Predicting Curves"):
        if epid < test_epoch - 1:
            ids_in_batch = np.arange(test_batch_size * epid, test_batch_size * (epid + 1))
        else:
            ids_in_batch = np.arange(test_batch_size * epid, test_batch_size * epid + num_images % test_batch_size)
        images = np.zeros((test_batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((test_batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)
        heights = np.zeros((test_batch_size, 1), dtype=np.float32)
        pitches = np.zeros((test_batch_size, 1), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        for bind in range(len(ids_in_batch)):
            db_ind = ids_in_batch[bind]
            item          = db.detections(db_ind)
            gt_cam_pitch  = item['gt_camera_pitch']
            gt_cam_height = item['gt_camera_height']
            image_file    = db.image_file(db_ind)
            image         = cv2.imread(image_file)
            height, width = image.shape[0:2]
            pad_image = image.copy()
            pad_mask = np.zeros((height, width, 1), dtype=np.float32)
            resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
            resized_mask = cv2.resize(pad_mask, (input_size[1], input_size[0]))
            masks[bind][0] = resized_mask.squeeze()
            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)
            resized_image = resized_image.transpose(2, 0, 1)
            images[bind] = resized_image
            heights[bind] = gt_cam_height
            pitches[bind] = gt_cam_pitch
        images        = torch.from_numpy(images)
        masks         = torch.from_numpy(masks)
        heights       = torch.from_numpy(heights)
        pitches       = torch.from_numpy(pitches)
        t0            = time.time()
        outputs, _    = nnet.test([images, masks, heights, pitches])
        t             = time.time() - t0

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if evaluator is not None:
            for bind in range(len(ids_in_batch)):
                db_ind = ids_in_batch[bind]
                evaluator.add_prediction(db_ind, results[bind].unsqueeze(0).cpu().numpy(), t)
    exp_name = 'apollosim'
    evaluator.exp_name = exp_name
    eval_stat = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))
    print(eval_stat)

    return eval_stat

def testing(db, nnet, result_dir, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug, evaluator=evaluator)