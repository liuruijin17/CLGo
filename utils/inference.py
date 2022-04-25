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

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1)) # b nj w*h
    idx = np.argmax(heatmaps_reshaped, 2) # b nj
    maxvals = np.amax(heatmaps_reshaped, 2) # b nj

    maxvals = maxvals.reshape((batch_size, num_joints, 1)) # b nj 1
    idx = idx.reshape((batch_size, num_joints, 1)) # b nj 1

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32) # b nj 2

    preds[:, :, 0] = (preds[:, :, 0]) % width # b nj
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) # b nj

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2)) # b nj 2
    pred_mask = pred_mask.astype(np.float32) # b nj 2

    preds *= pred_mask # preds = preds * pred_mask, b nj 2
    return preds, maxvals # b nj 2, b nj 1
    # heatmaps` int coord transform to preds` float coord


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    print(coords.shape)  # 3, 5, 2
    print(maxvals.shape) # 3, 5, 1
    exit()

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]): # batch
        for p in range(coords.shape[1]): # num_jointss
            hm = batch_heatmaps[n][p] # each image each joint
            px = int(math.floor(coords[n][p][0] + 0.5)) # x
            py = int(math.floor(coords[n][p][1] + 0.5)) # y
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]): # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
    # float coords transform to target_coords

def affine_final_preds(quan_res, batch_joints, center, scale):

    quan_height = quan_res[0]
    quan_width  = quan_res[1]
    batch_joints = batch_joints.copy()
    batch_joints[:, :, 0] *= quan_width
    batch_joints[:, :, 1] *= quan_height

    preds = batch_joints.copy()
    # Transform back
    for i in range(batch_joints.shape[0]):  # batch
        preds[i] = transform_preds(batch_joints[i],
                                   center[i],
                                   scale[i],
                                   [quan_width, quan_height])

    return preds




