# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torchvision
import cv2

from utils.inference import get_max_preds

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)
BLACK = (0, 0, 0)

# SBC_colors = [ORANGE, RED, CYAN, DARK_GREEN, GREEN, BLUE, YELLOW, PURPLE, PINK]
SBC_colors = [ORANGE, ORANGE, ORANGE, RED, RED, RED, CYAN, CYAN, CYAN]

KPS_colors = [DARK_GREEN, DARK_GREEN, YELLOW, YELLOW, PINK]

subclasses = [BLACK, ORANGE, CYAN, PINK, DARK_GREEN, RED]

def save_batch_images(batch_image,
                      batch_boxes,
                      batch_labels,
                      file_name,
                      db,
                      nrow=2,
                      padding=2):
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_boxes[k]
            labels = batch_labels[k]
            num_box = boxes.shape[0]
            boxes = boxes[labels == 1]
            labels = labels[labels == 1]
            i = 0
            for n in range(num_box):
                # lane = boxes[:, 3:][n]
                lane = boxes[n]
                seq_len = (len(lane) - 5) // 8
                xs = lane[3:3+seq_len]
                ys = lane[3+seq_len:3+seq_len*2]
                ys = ys[xs >= 0] * H
                xs = xs[xs >= 0] * W
                for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                    j_x = x * width + padding + xcoord
                    j_y = y * height + padding + ycoord
                    cv2.circle(ndarr, (int(j_x), int(j_y)), 2, BLUE, 10)
                i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_preds(batch_image,
                                batch_boxes,
                                batch_labels,
                                file_name,
                                db,
                                pitches,
                                heights,
                                nrow=2,
                                padding=2):

    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            pred          = batch_boxes[k].cpu().numpy()
            labels        = batch_labels[k].cpu().numpy()
            gt_cam_height = heights[k][0].cpu().numpy()
            gt_cam_pitch  = pitches[k][0].cpu().numpy()
            pred          = pred[labels == 1]
            labels        = labels[labels == 1]
            num_pred      = pred.shape[0]
            if num_pred > 0:
                for n, lane in enumerate(pred):
                    lane = lane[1:]
                    lower = np.minimum(lane[2], lane[3])  # gflat
                    upper = np.maximum(lane[2], lane[3])  # gflat
                    lane = lane[4:4+4]  # gflat
                    ys = np.linspace(lower, upper, num=100)  # gflat
                    xs = np.polyval(lane, ys)  # gflat
                    ys = ys*db.gflatYnorm
                    xs = xs*db.gflatXnorm
                    x_2d = 2015*xs + 960*np.cos(gt_cam_pitch)*ys
                    y_2d = (2015*-np.sin(gt_cam_pitch)+540*np.cos(gt_cam_pitch))*ys + 2015*gt_cam_height
                    h_2d = np.cos(gt_cam_pitch)*ys
                    x_2d = x_2d / h_2d
                    y_2d = y_2d / h_2d
                    x_2d = x_2d / 1920.
                    y_2d = y_2d / 1080.
                    points = np.zeros((len(ys), 2), dtype=np.int32)
                    points[:, 1] = (y_2d*H).astype(int)
                    points[:, 0] = (x_2d*W).astype(int)
                    points = points[(points[:, 0] > 0) & (points[:, 0] < W)]
                    points[:, 0] += x * width + padding
                    points[:, 1] += y * height + padding
                    for current_point, next_point in zip(points[:-1], points[1:]):
                        cv2.line(ndarr, tuple(current_point), tuple(next_point), color=RED, thickness=2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_ipms(batch_image,
                               batch_boxes,
                               batch_labels,
                               file_name,
                               db,
                               tgt_pitches,
                               tgt_heights,
                               tgt_flag,
                               nrow=2,
                               padding=2):
    np_image = batch_image.permute(0, 2, 3, 1).cpu().numpy()
    K = db.K
    H_g2ipm = np.linalg.inv(db.H_ipm2g)
    aug_mat = np.identity(3, dtype=np.float)
    ipm_image = []
    for i in range(np_image.shape[0]):
        img = np_image[i]
        ipm_canvas = (img - np.min(img)) / (np.max(img) - np.min(img))
        H_g2im = db.homograpthy_g2im(tgt_pitches[i][0].cpu().numpy(), tgt_heights[i][0].cpu().numpy(), K)
        H_im2ipm = np.linalg.inv(np.matmul(db.H_crop_ipm, np.matmul(H_g2im, db.H_ipm2g)))
        H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))
        im_ipm = cv2.warpPerspective(ipm_canvas, H_im2ipm, (db.ipm_w, db.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)
        ipm_image.append(im_ipm)
    batch_image = torch.from_numpy(np.stack(ipm_image, axis=0)).permute(0, 3, 1, 2)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_boxes[k]
            labels = batch_labels[k]
            flags = tgt_flag[k]
            num_box = boxes.shape[0]
            boxes = boxes[labels == 1]
            labels = labels[labels == 1]
            i = 0
            for n in range(num_box):
                # lane = boxes[:, 3:][n]
                lane = boxes[n]
                seq_len = (len(lane) - 5) // 8
                xs = lane[5+seq_len*5:5+seq_len*6] * db.gflatXnorm
                ys = lane[5+seq_len*6:5+seq_len*7] * db.gflatYnorm
                xs = xs[flags[n] > 0]
                ys = ys[flags[n] > 0]
                xs, ys = db.homographic_transformation(H_g2ipm, xs.cpu().numpy(), ys.cpu().numpy())
                for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                    j_x = x * width + padding + xcoord
                    j_y = y * height + padding + ycoord
                    cv2.circle(ndarr, (int(j_x), int(j_y)), 2, BLUE, 2)
                i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_pred_ipms(batch_image,
                                    batch_boxes,
                                    batch_labels,
                                    file_name,
                                    db,
                                    tgt_pitches,
                                    tgt_heights,
                                    nrow=2,
                                    padding=2):
    np_image = batch_image.permute(0, 2, 3, 1).cpu().numpy()
    K = db.K
    H_g2ipm = np.linalg.inv(db.H_ipm2g)
    aug_mat = np.identity(3, dtype=np.float)
    ipm_image = []
    for i in range(np_image.shape[0]):
        img = np_image[i]
        ipm_canvas = (img - np.min(img)) / (np.max(img) - np.min(img))
        H_g2im = db.homograpthy_g2im(tgt_pitches[i][0].cpu().numpy(), tgt_heights[i][0].cpu().numpy(), K)
        H_im2ipm = np.linalg.inv(np.matmul(db.H_crop_ipm, np.matmul(H_g2im, db.H_ipm2g)))
        H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))
        im_ipm = cv2.warpPerspective(ipm_canvas, H_im2ipm, (db.ipm_w, db.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)
        ipm_image.append(im_ipm)
        # ipm_laneline = im_ipm.copy()
    batch_image = torch.from_numpy(np.stack(ipm_image, axis=0)).permute(0, 3, 1, 2)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            pred = batch_boxes[k].cpu().numpy()
            labels = batch_labels[k].cpu().numpy()
            gt_cam_height = tgt_heights[k][0].cpu().numpy()
            gt_cam_pitch = tgt_pitches[k][0].cpu().numpy()
            pred = pred[labels == 1]
            num_pred = pred.shape[0]
            labels = labels[labels == 1]
            if num_pred > 0:
                for n, lane in enumerate(pred):
                    lane  = lane[1:]
                    lower = np.minimum(lane[2], lane[3])
                    upper = np.maximum(lane[2], lane[3])
                    zslane = lane[4+4:4+4+4]
                    lane  = lane[4:4+4]
                    ys    = np.linspace(lower, upper, num=100)
                    xs    = np.polyval(lane, ys)
                    zs    = np.polyval(zslane, ys)
                    zs    = zs * db.gflatZnorm
                    ys    = ys * db.gflatYnorm
                    xs    = xs * db.gflatXnorm

                    xs = xs * gt_cam_height / (gt_cam_height - zs)
                    ys = ys * gt_cam_height / (gt_cam_height - zs)

                    valid_indices = np.logical_and(np.logical_and(ys > 0, ys < 200),
                                                   np.logical_and(xs > 3 * db.x_min, xs < 3 * db.x_max))
                    ground_xs = xs[valid_indices]
                    ground_ys = ys[valid_indices]
                    if ground_xs.shape[0] < 2 or np.sum(
                            np.logical_and(ground_xs > db.x_min, ground_xs < db.x_max)) < 2:
                        continue
                    xs, ys = db.homographic_transformation(H_g2ipm, ground_xs, ground_ys)
                    points = np.zeros((len(ys), 2), dtype=np.int32)
                    points[:, 1] = ys
                    points[:, 0] = xs
                    points[:, 0] += x * width + padding
                    points[:, 1] += y * height + padding
                    for current_point, next_point in zip(points[:-1], points[1:]):
                        cv2.line(ndarr, tuple(current_point), tuple(next_point), color=RED, thickness=2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_debug_images_lstr3d(input, tgt_boxes, tgt_class,
                            pred_boxes, pred_class, prefix=None,
                            db=None, tgt_pitches=None, tgt_heights=None, tgt_flag=None):
    save_batch_images(
        input, tgt_boxes, tgt_class,
        '{}_img_gt.jpg'.format(prefix), db)

    save_batch_image_with_preds(
        input, pred_boxes, pred_class,
        '{}_img_pred.jpg'.format(prefix), db,
        tgt_pitches, tgt_heights)

    save_batch_image_with_ipms(
        input, tgt_boxes, tgt_class,
        '{}_ipm_gt.jpg'.format(prefix), db,
        tgt_pitches, tgt_heights, tgt_flag)

    save_batch_image_with_pred_ipms(
        input, pred_boxes, pred_class,
        '{}_ipm_pred.jpg'.format(prefix), db,
        tgt_pitches, tgt_heights)



