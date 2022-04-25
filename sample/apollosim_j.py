import cv2
import math
import numpy as np
import torch
import random
import string
from copy import deepcopy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_, get_affine_transform, affine_transform
from .utils import *

def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    # rot = random.uniform(-10, 10)
    center_x = img.shape[1] / 2
    center_y = img.shape[0] / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    # img_rot = np.array(img)
    img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    # img_rot = img.rotate(rot)
    # rot = rot / 180 * np.pi
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat

def kp_detection(db, k_ind, lane_debug=False):
    data_rng     = system_configs.data_rng
    batch_size   = system_configs.batch_size
    input_size   = db.configs["input_size"] # [h w]
    lighting     = db.configs["lighting"] # true
    rand_color   = db.configs["rand_color"] # color
    # allocating memory
    images   = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    masks    = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W
    heights  = np.zeros((batch_size, 1), dtype=np.float32)
    pitches  = np.zeros((batch_size, 1), dtype=np.float32)
    gt_lanes = []
    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if k_ind == 0:
            db.shuffle_inds()
        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading ground truth
        item  = db.detections(db_ind) # all in the raw coordinate
        img   = cv2.imread(item['path'])  # h w c
        mask  = np.ones((1, input_size[0], input_size[1], 1), dtype=np.bool)
        gt_2dgflatlabels = item['gt_2dgflatlabels']
        gt_2dgflatflags  = item['gt_2dgflatflags']
        gt_camera_pitch  = item['gt_camera_pitch']
        gt_camera_height = item['gt_camera_height']

        K = db.K
        P_g2im = db.projection_g2im(gt_camera_pitch, gt_camera_height, K)  # used for x=PX (3D to 2D)
        H_g2im = db.homograpthy_g2im(gt_camera_pitch, gt_camera_height, K)
        H_im2g = np.linalg.inv(H_g2im)
        P_g2gflat = np.matmul(H_im2g, P_g2im)  # convert 3d lanes on the ground
        aug_mat = np.identity(3, dtype=np.float)
        img = img[db.h_crop:db.h_org - db.h_crop, 0:db.w_org, :]
        img = cv2.resize(img, (db.w_net, db.h_net))  #  360 480
        # img, aug_mat = data_aug_rotate(img)
        heights[b_ind] = gt_camera_height
        pitches[b_ind] = gt_camera_pitch

        # lane_debug = True
        if lane_debug:
            draw_2dgflatlabels      = deepcopy(gt_2dgflatlabels)
            draw_2dgflatflags       = deepcopy(gt_2dgflatflags)
            draw_camera_height      = deepcopy(gt_camera_height)

        # clip polys
        gt_2dgflatflags  = gt_2dgflatflags[gt_2dgflatlabels[:, 0] > 0]
        gt_2dgflatlabels = gt_2dgflatlabels[gt_2dgflatlabels[:, 0] > 0]
        gt_camera_pitch  = np.stack([gt_camera_pitch] * len(gt_2dgflatlabels), axis=-1)
        gt_camera_height = np.stack([gt_camera_height] * len(gt_2dgflatlabels), axis=-1)

        # Repeat batch_size times to satisfy the chunksize
        gt_2dgflatlabels = np.stack([gt_2dgflatlabels] * batch_size, axis=0)
        gt_2dgflatflags  = np.stack([gt_2dgflatflags] * batch_size, axis=0)
        gt_camera_pitch  = np.stack([gt_camera_pitch] * batch_size, axis=0)
        gt_camera_height = np.stack([gt_camera_height] * batch_size, axis=0)
        gt_lanes.append(torch.from_numpy(gt_2dgflatlabels.astype(np.float32)))
        gt_lanes.append(torch.from_numpy(gt_2dgflatflags.astype(np.float32)))
        gt_lanes.append(torch.from_numpy(gt_camera_height.astype(np.float32)))
        gt_lanes.append(torch.from_numpy(gt_camera_pitch.astype(np.float32)))

        img = (img / 255.).astype(np.float32)
        if rand_color:
            color_jittering_(data_rng, img)
            if lighting:
                lighting_(data_rng, img, 0.1, db.eig_val, db.eig_vec)
        normalize_(img, db.mean, db.std)
        images[b_ind] = img.transpose((2, 0, 1))
        masks[b_ind]  = np.logical_not(mask[:, :, :, 0])

        # debug
        if lane_debug:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            ipm_canvas = deepcopy(img)
            img_h, img_w, _ = img.shape

            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')

            H_im2ipm = np.linalg.inv(np.matmul(db.H_crop_ipm, np.matmul(H_g2im, db.H_ipm2g)))
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))
            im_ipm = cv2.warpPerspective(ipm_canvas, H_im2ipm, (db.ipm_w, db.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)
            ipm_laneline = im_ipm.copy()
            H_g2ipm = np.linalg.inv(db.H_ipm2g)

            # Draw image-view label (2D)
            for i, lane in enumerate(draw_2dgflatlabels):
                if lane[0] == 0:  # Skip invalid lanes
                    continue
                # lane = lane[3:]  # remove conf, upper and lower positions
                seq_len = (len(lane)-5) // 5
                gt_2dlane = lane[:1+2+seq_len*2]
                gt_gflatlane = lane[1+2+seq_len*2:]
                lane = gt_2dlane[3:]
                xs = lane[:seq_len][draw_2dgflatflags[i] > 0]
                ys = lane[seq_len:seq_len*2][draw_2dgflatflags[i] > 0]
                ys = ys[xs >= 0]
                xs = xs[xs >= 0]
                for p in zip(xs, ys):
                    p = (int(p[0] * img_w), int(p[1] * img_h))
                    img = cv2.circle(img, p, 5, color=(0, 0, 255), thickness=-1)
                cv2.putText(img, str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))

                gflatlane = gt_gflatlane[2:]
                gflatXs = gflatlane[:seq_len][draw_2dgflatflags[i] > 0] * db.gflatXnorm
                # print('gflatXs: {}'.format(gflatXs))
                gflatYs = gflatlane[seq_len:seq_len*2][draw_2dgflatflags[i] > 0] * db.gflatYnorm
                # print('gflatYs: {}'.format(gflatYs))
                gflatZs = gflatlane[seq_len*2:][draw_2dgflatflags[i] > 0] * db.gflatZnorm
                x_ipm, y_ipm = db.homographic_transformation(H_g2ipm, gflatXs, gflatYs)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_ipm.shape[0]):
                    ipm_laneline = cv2.line(ipm_laneline, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), [0, 0, 1], 1)

                XS, YS = db.transform_lane_gflat2g(draw_camera_height, gflatXs, gflatYs, gflatZs)
                ax3.plot(XS, YS, gflatZs, color=[0, 0, 1])
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            bottom, top = ax3.get_zlim()
            # print(bottom, top)
            ax3.set_xlim(-30, 30)
            ax3.set_ylim(0, 200)
            ax3.set_zlim(min(bottom, -1), max(top, 1))
            ax1.imshow(img)
            ax2.imshow(ipm_laneline)
            plt.show()
            exit()

    images   = torch.from_numpy(images)
    masks    = torch.from_numpy(masks)
    heights  = torch.from_numpy(heights)
    pitches  = torch.from_numpy(pitches)

    return {
               "xs": [images, masks, heights, pitches],
               "ys": [images, *gt_lanes]
           }, k_ind


def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)


