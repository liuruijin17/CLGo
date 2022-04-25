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
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        # print(prob.shape)
        # print(prob)
        # exit()
        scores, labels = prob.max(-1)
        labels[labels != 1] = 0
        # results = torch.cat([labels.unsqueeze(-1).float(), out_bbox], dim=-1)
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

    for ind in tqdm(range(0, num_images), ncols=60, desc="locating kps"):
        db_ind        = db_inds[ind]
        # image_id      = db.image_ids(db_ind)
        item          = db.detections(db_ind)
        gt_cam_pitch  = item['gt_camera_pitch']
        gt_cam_height = item['gt_camera_height']
        image_file    = db.image_file(db_ind)
        image         = cv2.imread(image_file)
        height, width = image.shape[0:2]

        images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
        masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
        heights = np.zeros((1, 1), dtype=np.float32)
        pitches = np.zeros((1, 1), dtype=np.float32)
        orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
        pad_image     = image.copy()
        pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
        resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
        if debug:
            canvas = deepcopy(resized_image)
        resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
        masks[0][0]   = resized_mask.squeeze()
        resized_image = resized_image / 255.
        normalize_(resized_image, db.mean, db.std)
        resized_image = resized_image.transpose(2, 0, 1)
        images[0]     = resized_image
        heights[0]    = gt_cam_height
        pitches[0]    = gt_cam_pitch
        images        = torch.from_numpy(images)
        masks         = torch.from_numpy(masks)
        heights       = torch.from_numpy(heights)
        pitches       = torch.from_numpy(pitches)

        if attn_map:
            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            hooks = [
                nnet.model.module.layer4[-1].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)),
                nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])),
                nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1]))
            ]

        t0            = time.time()
        outputs, _    = nnet.test([images, masks, heights, pitches])
        t             = time.time() - t0

        if attn_map:
            for hook in hooks:
                hook.remove()
            conv_features    = conv_features[0]
            enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0]

        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if evaluator is not None:
            evaluator.add_prediction(ind, results.cpu().numpy(), t)

        if debug:
            img_lst = image_file.split('/')
            lane_debug_dir = os.path.join(result_dir, "lane_debug")
            if not os.path.exists(lane_debug_dir):
                os.makedirs(lane_debug_dir)

            # Draw ipm-view
            # img_canvas, ipm_laneline = db.draw_annotation(ind, pred=results[0].cpu().numpy(), cls_pred=None, img=canvas)
            img_canvas, ipm_laneline = db.draw_annotation(ind, pred=results[0].cpu().numpy(), cls_pred=None, img=canvas)
            # cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
            #                          + img_lst[-2] + '_'
            #                          + os.path.basename(image_file[:-4]) + '_IMG-VIEW.jpg'), img_canvas)
            #
            # cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
            #                          + img_lst[-2] + '_'
            #                          + os.path.basename(image_file[:-4]) + '_TOP-VIEW.jpg'), ipm_laneline)
            # exit()

            # # Draw 3d-view
            # pred_plt = db.draw_3dannotation(ind, pred=None, cls_pred=None, img=(img_canvas, ipm_laneline))
            pred_plt = db.draw_3dannotation(ind, pred=results[0].cpu().numpy(), cls_pred=None,
                                            img=(img_canvas, ipm_laneline))
            pred_plt.savefig(os.path.join(lane_debug_dir, img_lst[-3] + '_' + img_lst[-2] + '_'
                                          + os.path.basename(image_file[:-4]) + '_3D.jpg'))
            pred_plt.close()
            # exit()

    if not debug:
        exp_name = 'apollosim'
        evaluator.exp_name = exp_name
        eval_stat = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))
        # print(eval_stat)

    return eval_stat

def testing(db, nnet, result_dir, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug, evaluator=evaluator)