import sys
import json
import os
import numpy as np
import pickle
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from copy import deepcopy
from scipy.interpolate import interp1d

import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from db.detection import DETECTION
from config import system_configs

from db.tools import eval_3D_lane

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)

GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [RED, GREEN, DARK_GREEN, PURPLE, CHOCOLATE, PEACHPUFF, STATEGRAY]
PRED_HIT_COLOR = GREEN
PRED_MISS_COLOR = RED
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class APOLLOSIM(DETECTION):
    def __init__(self, db_config, split, is_eval=False, is_resample=True, is_predcam=False):
        super(APOLLOSIM, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        # result_dir = system_configs.result_dir
        cache_dir   = system_configs.cache_dir
        max_lanes   = system_configs.max_lanes
        self.metric = 'default'
        self.is_resample = is_resample
        print('is_resample: {}'.format(is_resample))
        inp_h, inp_w = db_config['input_size']

        # define image pre-processor
        # self.totensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)
        # self.data_aug = data_aug  # False

        # dataset parameters
        # dataset_name = 'standard'  # illus_chg/rare_subset/standard
        self.dataset_name = system_configs.dataset_name  # illus_chg
        self.no_3d = False
        self.no_centerline = True

        self.h_org  = 1080
        self.w_org  = 1920
        self.org_h  = 1080
        self.org_w  = 1920
        self.h_crop = 0
        self.crop_y = 0

        # parameters related to service network
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        self.ipm_h = 208
        self.ipm_w = 128
        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.K = np.array([[2015., 0., 960.], [0., 2015., 540.], [0., 0., 1.]])
        self.H_crop_ipm = self.homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_net, self.w_net])
        self.H_crop_im  = self.homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_org, self.w_org])
        # org2resized+cropped
        self.H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]),
            np.float32(self.top_view_region))
        self.fix_cam = False

        x_min = self.top_view_region[0, 0]  # -10
        x_max = self.top_view_region[1, 0]  # 10
        self.x_min = x_min  # -10
        self.x_max = x_max  # 10
        self.anchor_y_steps = [  5,  10,  15,  20,  30,  40,  50,  60,  80,  100]
        # self.anchor_y_steps = [   5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        #                          15, 16, 17, 18, 19, 20, 22, 24, 26, 28,
        #                          32, 36, 40, 48, 56, 60, 80, 100]
        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        if self.is_resample:
            self.gflatYnorm = self.anchor_y_steps[-1]
            self.gflatZnorm = 10
            self.gflatXnorm = 30
        else:
            self.gflatYnorm = 200
            self.gflatZnorm = 1
            self.gflatXnorm = 20

        self.pitch = 3  # pitch angle of camera to ground in centi degree
        self.cam_height = 1.55  # height of camera in meters
        self.batch_size = system_configs.batch_size

        if self.no_centerline:  # False
            self.num_types = 1
        else:
            self.num_types = 3
        if self.is_resample:
            self.sample_hz = 1
        else:
            self.sample_hz = 4

        self._split = split
        self._dataset = {
            "train": ['train'],
            "test": ['test'],
            # "sub_train": ['sub_train'],
            # "validation": ['validation']
        }[self._split]

        self.root = os.path.join(data_dir, 'Apollo_Sim_3D_Lane_Release')
        data_dir = os.path.join(self.root, 'data_splits', self.dataset_name)
        if self.root is None:
            raise Exception('Please specify the root directory')
        self.img_w, self.img_h = self.h_org, self.w_org  # apollo sim original image resolution
        self.max_2dlanes     = 0
        self.max_gflatlanes  = 0
        self.max_3dlanes     = 0
        self.max_2dpoints    = 0
        self.max_gflatpoints = 0
        self.max_3dpoints    = 0
        self.X3d, self.Y3d, self.Z3d = [0, 0], [0, 0], [0, 0]
        self.Xgflat, self.Ygflat = [0, 0], [0, 0]
        self.normalize = True
        self.to_tensor = ToTensor()

        self.aug_chance = 0.9090909090909091
        self._image_file = []

        self.augmentations = [{'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
                              {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
                              {'name': 'CropToFixedSize', 'parameters': {'height': 972, 'width': 1728}}]


        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # if max_lanes is not None:
        #     self.max_lanes = max_lanes
        if 'train' in self._dataset:
            self.anno_files = [os.path.join(data_dir, path + '.json') for path in self._dataset]
        else:
            self.anno_files = [os.path.join(data_dir, path + '.json') for path in ['test']]


        self._data = "apollosim_j"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self._cat_ids = [
            0
        ]  # 0 car
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "apollosim_{}.pkl".format(self._dataset))


        if self.augmentations is not None:
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in self.augmentations]  # add augmentation

        transformations = iaa.Sequential([Resize({'height': inp_h, 'width': inp_w})])
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=self.aug_chance), transformations])

        if is_eval:
            if is_predcam:
                raise NotImplementedError
                # self._load_predcam_data(result_path=result_path)
            else:
                self._load_eval_data()
        else:
            self._load_data()

        self._db_inds = np.arange(len(self._image_ids))

    def _load_data(self, debug_lane=False):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            self._transform_annotations()

            if debug_lane:
                pass
            else:
                with open(self._cache_file, "wb") as f:
                    pickle.dump([self._annotations,
                                 self._image_ids,
                                 self._image_file,
                                 self.max_2dlanes, self.max_3dlanes, self.max_gflatlanes,
                                 self.max_2dpoints, self.max_3dpoints, self.max_gflatpoints,
                                 self.X3d, self.Y3d, self.Z3d,
                                 self.Xgflat, self.Ygflat], f)
        else:
            with open(self._cache_file, "rb") as f:
                (self._annotations,
                 self._image_ids,
                 self._image_file,
                 self.max_2dlanes, self.max_3dlanes, self.max_gflatlanes,
                 self.max_2dpoints, self.max_3dpoints, self.max_gflatpoints,
                 self.X3d, self.Y3d, self.Z3d,
                 self.Xgflat, self.Ygflat) = pickle.load(f)
        assert self.max_2dlanes == self.max_3dlanes
        assert self.max_3dlanes == self.max_gflatlanes
        assert self.max_2dpoints == self.max_3dpoints
        assert self.max_3dpoints == self.max_gflatpoints

        print('{}.max_2dlanes: {}\n'
              '{}.max_3dlanes: {}\n'
              '{}.max_gflatlanes: {}\n'
              '{}.max_2dpoints: {}\n'
              '{}.max_3dpoints: {}\n'
              '{}.max_gflatpoints: {}\n'
              '{}.X3d: {}\n'
              '{}.Y3d: {}\n'
              '{}.Z3d: {}\n'
              '{}.Xgflat: {}\n'
              '{}.Ygflat: {}'.format(self.dataset_name, self.max_2dlanes,
                                     self.dataset_name, self.max_3dlanes,
                                     self.dataset_name, self.max_gflatlanes,
                                     self.dataset_name, self.max_2dpoints,
                                     self.dataset_name, self.max_3dpoints,
                                     self.dataset_name, self.max_gflatpoints,
                                     self.dataset_name, self.X3d,
                                     self.dataset_name, self.Y3d,
                                     self.dataset_name, self.Z3d,
                                     self.dataset_name, self.Xgflat,
                                     self.dataset_name, self.Ygflat))

    def _extract_data(self):
        image_id  = 0
        max_2dlanes, max_3dlanes, max_gflatlanes = 0, 0, 0
        self._old_annotations = {}
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                for line in anno_obj:
                    info_dict = json.loads(line)
                    # dict_keys(['raw_file', 'cam_height', 'cam_pitch',
                    # 'centerLines', 'laneLines', 'centerLines_visibility', 'laneLines_visibility'])
                    gt_lane_pts = info_dict['laneLines']
                    if len(gt_lane_pts) < 1:
                        continue
                    gt_lane_visibility = info_dict['laneLines_visibility']
                    image_path = os.path.join(self.root, info_dict['raw_file'])
                    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)

                    # if not self.fix_cam:
                    gt_cam_height = info_dict['cam_height']
                    gt_cam_pitch = info_dict['cam_pitch']
                    P_g2im = self.projection_g2im(gt_cam_pitch, gt_cam_height, self.K)  # used for x=PX (3D to 2D)
                    H_g2im = self.homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
                    H_im2g = np.linalg.inv(H_g2im)
                    P_g2gflat = np.matmul(H_im2g, P_g2im)
                    aug_mat = np.identity(3, dtype=np.float)

                    gt_lanes = []
                    # org_gt_lanes = []
                    for i, lane in enumerate(gt_lane_pts):
                        # A GT lane can be either 2D or 3D
                        # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                        closest_point  = lane[0]
                        remotest_point = lane[-1]
                        sampled_points = lane[1:-1:self.sample_hz]
                        sampled_points.insert(0, closest_point)
                        sampled_points.append(remotest_point)
                        lane = np.array(sampled_points)
                        # lane = np.array(lane[::self.sample_hz])

                        closest_viz  = gt_lane_visibility[i][0]
                        remotest_viz = gt_lane_visibility[i][-1]
                        sampled_viz  = gt_lane_visibility[i][1:-1:self.sample_hz]
                        sampled_viz.insert(0, closest_viz)
                        sampled_viz.append(remotest_viz)
                        lane_visibility = np.array(sampled_viz)
                        # lane_visibility = np.array(gt_lane_visibility[i][::self.sample_hz])
                        # prune gt lanes by visibility labels
                        pruned_lane = self.prune_3d_lane_by_visibility(lane, lane_visibility)
                        # prune out-of-range points are necessary before transformation -30~30
                        pruned_lane = self.prune_3d_lane_by_range(pruned_lane, 3*self.x_min, 3*self.x_max)

                        # Resample
                        if self.is_resample:
                            if pruned_lane.shape[0] < 2:
                                continue
                                # Above code resample 3D points
                                # print(pruned_lane.shape)
                            pruned_lane = self.make_lane_y_mono_inc(pruned_lane)
                            # print(pruned_lane.shape)
                            if pruned_lane.shape[0] < 2:
                                continue
                            x_values, z_values, visibility_vec = self.resample_laneline_in_y(pruned_lane,
                                                                                             self.anchor_y_steps,
                                                                                             out_vis=True)
                            x_values = x_values[visibility_vec]
                            z_values = z_values[visibility_vec]
                            y_values = np.array(self.anchor_y_steps)[visibility_vec]
                            pruned_lane = np.stack([x_values, y_values, z_values], axis=-1)
                            # print(pruned_lane.shape);exit()

                        if pruned_lane.shape[0] > 1:
                            gt_lanes.append(pruned_lane)

                    # save the gt 3d lanes
                    gt_3dlanes = deepcopy(gt_lanes)

                    # convert 3d lanes to flat ground space  x_bar y_bar Z (meter i think)
                    self.convert_lanes_3d_to_gflat(gt_lanes, P_g2gflat)

                    gflatlanes = []
                    real_gt_3dlanes = []
                    for i in range(len(gt_lanes)):
                        gflatlane = gt_lanes[i]
                        gt_3dlane = gt_3dlanes[i]
                        valid_indices = np.logical_and(np.logical_and(gflatlane[:, 1] > 0, gflatlane[:, 1] < 200),
                                                       np.logical_and(gflatlane[:, 0] > 3 * self.x_min, gflatlane[:, 0] < 3 * self.x_max))
                        gflatlane = gflatlane[valid_indices, ...]
                        gt_3dlane = gt_3dlane[valid_indices, ...]
                        if gflatlane.shape[0] < 2 or np.sum(np.logical_and(gflatlane[:, 0] > self.x_min, gflatlane[:, 0] < self.x_max)) < 2:
                            continue
                        gflatlanes.append(gflatlane)
                        real_gt_3dlanes.append(gt_3dlane)

                    P_gt = np.matmul(self.H_crop_im, H_g2im)
                    P_gt = np.matmul(aug_mat, P_gt)
                    lanes = []
                    for i in range(len(gflatlanes)):
                        gflatlane = gflatlanes[i]
                        x_2d, y_2d = self.homographic_transformation(P_gt, gflatlane[:, 0], gflatlane[:, 1])
                        assert gflatlane.shape[0] == x_2d.shape[0]
                        assert x_2d.shape[0] == y_2d.shape[0]
                        # lanes.append([(x, y) for (x, y) in zip(x_2d, y_2d) if x >= 0])
                        lanes.append([(x, y) for (x, y) in zip(x_2d, y_2d)])

                    lanes = [lane for lane in lanes if len(lane) > 0]
                    if not len(lanes):
                        continue

                    self._image_file.append(image_path)
                    self._image_ids.append(image_id)

                    max_2dlanes       = max(max_2dlanes, len(lanes))
                    self.max_2dlanes  = max_2dlanes

                    max_gflatlanes      = max(max_gflatlanes, len(gflatlanes))
                    self.max_gflatlanes = max_gflatlanes

                    max_3dlanes      = max(max_3dlanes, len(real_gt_3dlanes))
                    self.max_3dlanes = max_3dlanes

                    self.max_2dpoints    = max(self.max_2dpoints, max([len(l) for l in lanes]))
                    self.max_gflatpoints = max(self.max_gflatpoints, max([len(l) for l in gflatlanes]))
                    self.max_3dpoints    = max(self.max_3dpoints, max([len(l) for l in real_gt_3dlanes]))

                    self.X3d[1] = max(self.X3d[1], max([np.max(l[:, 0]) for l in real_gt_3dlanes]))
                    self.X3d[0] = min(self.X3d[0], min([np.min(l[:, 0]) for l in real_gt_3dlanes]))
                    self.Y3d[1] = max(self.Y3d[1], max([np.max(l[:, 1]) for l in real_gt_3dlanes]))
                    self.Y3d[0] = min(self.Y3d[0], min([np.min(l[:, 1]) for l in real_gt_3dlanes]))
                    self.Z3d[1] = max(self.Z3d[1], max([np.max(l[:, 2]) for l in real_gt_3dlanes]))
                    self.Z3d[0] = min(self.Z3d[0], min([np.min(l[:, 2]) for l in real_gt_3dlanes]))
                    self.Xgflat[1] = max(self.Xgflat[1], max([np.max(l[:, 0]) for l in gflatlanes]))
                    self.Xgflat[0] = min(self.Xgflat[0], min([np.min(l[:, 0]) for l in gflatlanes]))
                    self.Ygflat[1] = max(self.Ygflat[1], max([np.max(l[:, 1]) for l in gflatlanes]))
                    self.Ygflat[0] = min(self.Ygflat[0], min([np.min(l[:, 1]) for l in gflatlanes]))

                    self._old_annotations[image_id] = {
                        'path': image_path,
                        'gt_2dlanes': lanes,
                        'gt_3dlanes': real_gt_3dlanes,
                        'gt_gflatlanes': gflatlanes,
                        'aug': False,
                        'relative_path': info_dict['raw_file'],
                        'gt_camera_pitch': gt_cam_pitch,
                        'gt_camera_height': gt_cam_height,
                        'json_line': info_dict,

                    }
                    image_id += 1

    def _transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self._get_img_heigth(anno['path'])
            img_w = self._get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        gt_2dlanes = anno['gt_2dlanes']
        gt_gflatlanes = anno['gt_gflatlanes']
        gt_3dlanes = anno['gt_3dlanes']

        assert len(gt_2dlanes) == len(gt_gflatlanes)
        assert len(gt_3dlanes) == len(gt_gflatlanes)

        categories = anno['categories'] if 'categories' in anno else [1] * len(gt_2dlanes)
        gt_2dlanes = zip(gt_2dlanes, categories)

        # 1+2+(2*self.max_2dpoints)+2+(2*self.max_2dpoints)+(3*self.max_2dpoints)
        # c|2d_1|2d_2|u_2d|v_2d|3d_1|3d_2|3d_X|3d_Y|3d_Z|gflat_X|gflat_Y|gflat_Z
        lanes       = np.ones((self.max_2dlanes, 1+2+self.max_2dpoints*2), dtype=np.float32) * -1e5
        lanes3d     = np.ones((self.max_2dlanes, 2+self.max_2dpoints*3), dtype=np.float32) * -1e5
        lanesgflat  = np.ones((self.max_2dlanes, self.max_2dpoints*3), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        laneflags = np.ones((self.max_2dlanes, self.max_2dpoints), dtype=np.float32) * -1e-5
        # old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(gt_2dlanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 1+2:1+2+len(xs)] = xs
            lanes[lane_pos, (1+2+self.max_2dpoints):(1+2+self.max_2dpoints+len(ys))] = ys
            laneflags[lane_pos, :len(xs)] = 1.

            gt_3dlane = gt_3dlanes[lane_pos]
            assert len(lane) == len(gt_3dlane)
            lower, upper = gt_3dlane[0][1], gt_3dlane[-1][1]
            Xs = np.array([p[0] for p in gt_3dlane]) / self.gflatXnorm
            Ys = np.array([p[1] for p in gt_3dlane]) / self.gflatYnorm
            Zs = np.array([p[2] for p in gt_3dlane]) / self.gflatZnorm
            lanes3d[lane_pos, 0] = lower / self.gflatYnorm
            lanes3d[lane_pos, 1] = upper / self.gflatYnorm
            lanes3d[lane_pos, 2:(2+len(Xs))] = Xs
            lanes3d[lane_pos, (2+self.max_3dpoints):(2+self.max_3dpoints+len(Ys))] = Ys
            lanes3d[lane_pos, (2+self.max_3dpoints*2):(2+self.max_3dpoints*2+len(Zs))] = Zs

            gflatlane = gt_gflatlanes[lane_pos]
            assert len(lane) == len(gflatlane)
            gflat_Xs = np.array([p[0] for p in gflatlane]) / self.gflatXnorm
            gflat_Ys = np.array([p[1] for p in gflatlane]) / self.gflatYnorm
            gflat_Zs = np.array([p[2] for p in gflatlane]) / self.gflatZnorm

            lanesgflat[lane_pos, :len(gflat_Xs)] = gflat_Xs
            lanesgflat[lane_pos, self.max_gflatpoints:(self.max_gflatpoints+len(gflat_Ys))] = gflat_Ys
            lanesgflat[lane_pos, self.max_gflatpoints*2:(self.max_gflatpoints*2+len(gflat_Ys))] = gflat_Zs

        lanes = np.concatenate([lanes, lanes3d, lanesgflat], axis=-1)

        new_anno = {
            'path': anno['path'],
            'gt_2dgflatlabels': lanes,
            'gt_2dgflatflags': laneflags,
            'old_anno': anno,
            'categories': [cat for _, cat in gt_2dlanes],
            'gt_camera_pitch': anno['gt_camera_pitch'],
            'gt_camera_height': anno['gt_camera_height'],
        }

        return new_anno

    def _transform_annotations(self):
        print('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_annotation(old_anno)

    def _load_eval_data(self):
        self._extact_eval_data()
        self._transform_eval_annotations()

    def _extact_eval_data(self):
        image_id  = 0
        self._old_annotations = {}
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                for line in anno_obj:
                    info_dict = json.loads(line)
                    # dict_keys(['raw_file', 'cam_height', 'cam_pitch',
                    # 'centerLines', 'laneLines', 'centerLines_visibility', 'laneLines_visibility'])
                    image_path = os.path.join(self.root, info_dict['raw_file'])
                    gt_cam_height = info_dict['cam_height']
                    gt_cam_pitch = info_dict['cam_pitch']
                    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
                    self._image_file.append(image_path)
                    self._image_ids.append(image_id)
                    self._old_annotations[image_id] = {
                        'path': image_path,
                        'aug': False,
                        'relative_path': info_dict['raw_file'],
                        'json_line': info_dict,
                        'gt_camera_pitch': gt_cam_pitch,
                        'gt_camera_height': gt_cam_height,
                    }
                    image_id += 1

    def _transform_eval_annotation(self, anno):
        new_anno = {
            'path': anno['path'],
            'old_anno': anno,
            'gt_camera_pitch': anno['gt_camera_pitch'],
            'gt_camera_height': anno['gt_camera_height'],
        }
        return new_anno

    def _transform_eval_annotations(self):
        print('Now transforming EVALEVALEVAL annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_eval_annotation(old_anno)

    def __getitem__(self, idx, transform=False):
        # I think this part is only used when testing

        item = self._annotations[idx]
        img = cv2.imread(item['path'])
        gt_2dflatlabels = item['gt_2dgflatlabels']
        gt_2dgflatflags = item['gt_2dgflatflags']
        gt_camera_pitch = item['gt_camera_pitch']
        gt_camera_height = item['gt_camera_height']

        if transform:
            raise NotImplementedError

        return (img, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, idx)

    def pred2lanes(self, path, pred, y_samples, camera_height):
        ys = np.array(y_samples) / self.gflatYnorm
        lanes = []
        probs = []
        for lane in pred:
            if lane[1] == 0:
                continue
            # pred_height = lane[-2]
            # pred_height = lane[-2]
            # pred_pitch  = lane[-1]
            lane_xsys = lane[6:6+4]
            lane_zsys = lane[10:10+4]
            X_pred = np.polyval(lane_xsys, ys) * self.gflatXnorm
            Z_pred = np.polyval(lane_zsys, ys) * self.gflatZnorm

            valid_indices = (ys > lane[4]) & (ys < lane[5])
            if np.sum(valid_indices) < 2:
                continue
            X_pred = X_pred[valid_indices]
            Y_pred = ys[valid_indices] * self.gflatYnorm
            Z_pred = Z_pred[valid_indices]
            # X_pred, Y_pred = self.transform_lane_gflat2g(camera_height, X_pred, Y_pred, Z_pred)
            lanes.append(np.stack([X_pred, Y_pred, Z_pred], axis=-1).tolist())
            probs.append(float(lane[0]))

        return lanes, probs

    def pred2apollosimformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        old_anno = self._annotations[idx]['old_anno']
        # path = old_anno['path']
        relative_path = old_anno['relative_path']
        json_line = old_anno['json_line']
        gt_camera_height = old_anno['gt_camera_height']
        gt_camera_pitch = old_anno['gt_camera_pitch']
        pred_cam_height = pred[0, -2]
        pred_cam_pitch = pred[0, -1]
        self.mae_height += np.abs(pred_cam_height - gt_camera_height)
        self.mae_pitch += np.abs(pred_cam_pitch - gt_camera_pitch)
        # print(gt_camera_height, gt_camera_pitch)
        # print(pred[:, -2:])
        # y_samples = self.anchor_y_steps
        # y_samples = list((np.linspace(0, 1., num=100) * 200.))
        y_samples = list((np.linspace(self.top_view_region[2, 1]/self.gflatYnorm, self.top_view_region[0, 1]/self.gflatYnorm, num=100) * self.gflatYnorm))
        pred_lanes, prob_lanes = self.pred2lanes(relative_path, pred, y_samples, gt_camera_height)
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        json_line["pred_cam_height"] = pred_cam_height
        json_line["pred_cam_pitch"]  = pred_cam_pitch
        return json_line

    def save_apollosim_predictions(self, predictions, runtimes, filename):
        self.mae_height = 0
        self.mae_pitch = 0
        with open(filename, 'w') as jsonFile:
            for idx in range(len(predictions)):
                json_line = self.pred2apollosimformat(idx, predictions[idx], runtimes[idx])
                json.dump(json_line, jsonFile)
                jsonFile.write('\n')
            print('Height(m):\t{}'.format(self.mae_height / len(predictions)))
            print('Pitch(o):\t{}'.format(self.mae_pitch / len(predictions) * 180 / np.pi))
            print('Pitch(rad):\t{}'.format(self.mae_pitch / len(predictions)))
        # exit()
        return self.mae_height / len(predictions), self.mae_pitch / len(predictions) * 180 / np.pi


    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        # raise NotImplementedError
        pred_filename = 'apollosim_{}_{}_predictions_{}.json'.format(self.dataset_name, self.split, label)
        pred_filename = os.path.join(exp_dir, pred_filename)
        he, pe = self.save_apollosim_predictions(predictions, runtimes, pred_filename)
        if self.metric == 'default':
            evaluator = eval_3D_lane.LaneEval(self)
            eval_stats_pr = evaluator.bench_one_submit_varying_probs(pred_filename, self.anno_files[0])
            max_f_prob = eval_stats_pr['max_F_prob_th']
            eval_stats = evaluator.bench_one_submit(pred_filename, self.anno_files[0], prob_th=max_f_prob)
            print("Metrics: F-score,    AP, x error (close), x error (far), z error (close), z error (far)")
            print("Laneline:{:.3}, {:.3},   {:.3},           {:.3},         {:.3},           {:.3}".format(
                eval_stats[0], eval_stats_pr['laneline_AP'], eval_stats[3], eval_stats[4], eval_stats[5], eval_stats[6]))
            result = {
                'AP':  eval_stats_pr['laneline_AP'],
                'F-score': eval_stats[0],
                'x error (close)': eval_stats[3],
                'x error (far)': eval_stats[4],
                'z error (close)': eval_stats[5],
                'z error (far)': eval_stats[6]
            }
            # print("Centerline:{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}".format(
            #     eval_stats_pr['centerline_AP'], eval_stats[7], eval_stats[10], eval_stats[11], eval_stats[12], eval_stats[13]))
        elif self.metric == 'ours':
            raise NotImplementedError
        if not only_metrics:
            filename = 'apollosim_{}_{}_eval_result_{}.json'.format(self.dataset_name, self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(result, out_file)
        return {'F-score': eval_stats[0], 'AP': eval_stats_pr['laneline_AP'],
                'EH': he, 'EP': pe}

    def detections(self, ind):
        image_id  = self._image_ids[ind]
        item      = self._annotations[image_id]
        return item

    def __len__(self):
        return len(self._annotations)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def class_name(self, cid):
        cat_id = self._classes[cid]
        return cat_id

    def _get_img_heigth(self, path):
        return 1080

    def _get_img_width(self, path):
        return 1920

    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            # raise NotImplementedError
            img, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, _ = \
                self.__getitem__(idx, transform=False)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            _, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, _ = \
                self.__getitem__(idx, transform=False)
            img = (img * 255).astype(np.uint8)

        # print('self.H_crop_ipm: {}'.format(self.H_crop_ipm))
        # print('self.H_crop_im: {}'.format(self.H_crop_im))
        # print('self.top_view_region: {}'.format(self.top_view_region))
        # print('self.H_ipm2g: {}'.format(self.H_ipm2g))

        img_h, img_w, _ = img.shape
        img_canvas = deepcopy(img)
        K = self.K
        aug_mat = np.identity(3, dtype=np.float)
        H_g2im = self.homograpthy_g2im(gt_camera_pitch, gt_camera_height, K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop_ipm, np.matmul(H_g2im, self.H_ipm2g)))
        H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))

        P_g2im = self.projection_g2im(gt_camera_pitch, gt_camera_height, self.K)  # used for x=PX (3D to 2D)
        # H_g2im = self.homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2g = np.linalg.inv(H_g2im)
        P_g2gflat = np.matmul(H_im2g, P_g2im)

        ipm_canvas = deepcopy(img)
        im_ipm = cv2.warpPerspective(ipm_canvas / 255., H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        ipm_laneline = im_ipm.copy()
        H_g2ipm = np.linalg.inv(self.H_ipm2g)
        for i, lane in enumerate(gt_2dflatlabels):
            # lane = lane[3:]  # remove conf, upper and lower positions
            seq_len = len(lane-5) // 8
            xs = lane[3:3+seq_len][gt_2dgflatflags[i] > 0]
            ys = lane[3+seq_len:3+seq_len*2][gt_2dgflatflags[i] > 0]
            ys = ys[xs >= 0].astype(np.int)
            xs = xs[xs >= 0].astype(np.int)
            # for p in zip(xs, ys):
            #     p = (int(p[0] * img_w), int(p[1] * img_h))
            #     img_canvas = cv2.circle(img_canvas, p, 5, color=(0, 0, 255), thickness=-1)
            for p in range(1, ys.shape[0]):
                img_canvas = cv2.line(img_canvas, (xs[p - 1], ys[p - 1]), (xs[p], ys[p]), [0, 0, 1], 2)

            gflatlane = lane[5+seq_len*5:]
            gflatXs = gflatlane[:seq_len][gt_2dgflatflags[i] > 0] * self.gflatXnorm
            gflatYs = gflatlane[seq_len:seq_len*2][gt_2dgflatflags[i] > 0] * self.gflatYnorm
            x_ipm, y_ipm = self.homographic_transformation(H_g2ipm, gflatXs, gflatYs)
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)
            for k in range(1, x_ipm.shape[0]):
                ipm_laneline = cv2.line(ipm_laneline, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), [0, 0, 1], 2)
                # ipm_laneline = cv2.circle(ipm_laneline, (x_ipm[k], y_ipm[k]), 5, color=(255, 0, 0), thickness=-1)
        ipm_laneline = (ipm_laneline * 255).astype(np.uint8)

        # cv2.imshow('fff', ipm_laneline)
        # cv2.waitKey(0)
        # exit()
        if pred is None:
            print('Why')
            return img_canvas, ipm_laneline

        P_gt = np.matmul(self.H_crop_im, H_g2im)
        P_gt = np.matmul(aug_mat, P_gt)
        pred = pred[pred[:, 1].astype(int) == 1]
        matches, accs, _ = self.get_metrics(pred, idx)
        for i, lane in enumerate(pred):
            lower, upper = lane[4], lane[5]
            zlane = lane[10:14]
            lane = lane[6:10]  # remove upper, lower positions
            ys = np.linspace(lower, upper, num=100)
            xs = np.polyval(lane, ys)
            zs = np.polyval(zlane, ys)
            pred_ys = ys * self.gflatYnorm
            pred_xs = xs * self.gflatXnorm
            pred_zs = zs * self.gflatZnorm
            pred_xs, pred_ys = self.projective_transformation(P_g2gflat, pred_xs, pred_ys, pred_zs)
            valid_indices = np.logical_and(np.logical_and(pred_ys > 0, pred_ys < 200),
                                           np.logical_and(pred_xs > 3 * self.x_min, pred_xs < 3 * self.x_max))
            pred_xs = pred_xs[valid_indices]
            pred_ys = pred_ys[valid_indices]
            if pred_xs.shape[0] < 2 or np.sum(np.logical_and(pred_xs > self.x_min, pred_xs < self.x_max)) < 2:
                continue
            pred_ipm_xs, pred_ipm_ys = self.homographic_transformation(H_g2ipm, pred_xs, pred_ys)
            pred_ipm_xs = pred_ipm_xs.astype(np.int)
            pred_ipm_ys = pred_ipm_ys.astype(np.int)
            for k in range(1, pred_ipm_xs.shape[0]):
                ipm_laneline = cv2.line(ipm_laneline, (pred_ipm_xs[k - 1], pred_ipm_ys[k - 1]),
                                        (pred_ipm_xs[k], pred_ipm_ys[k]),
                                        [255, 0, 0], 2)

            pred_x2d, pred_y2d = self.homographic_transformation(P_gt, pred_xs, pred_ys)
            pred_x2d = (pred_x2d * self.w_net / self.w_org).astype(np.int)
            pred_y2d = (pred_y2d * self.h_net / self.h_org).astype(np.int)
            for k in range(1, pred_x2d.shape[0]):
                img_canvas = cv2.line(img_canvas, (pred_x2d[k - 1], pred_y2d[k - 1]), (pred_x2d[k], pred_y2d[k]),
                                      [255, 0, 0], 2)

        return img_canvas, ipm_laneline

    def draw_annotation_predCameraPoses(self, idx, pred=None, img=None, cls_pred=None):
        if img is None:
            # raise NotImplementedError
            img, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, _ = \
                self.__getitem__(idx, transform=False)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            _, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, _ = \
                self.__getitem__(idx, transform=False)
            img = (img * 255).astype(np.uint8)

        print('gt_camera_pitch: {}'.format(gt_camera_pitch))
        print('gt_camera_height: {}'.format(gt_camera_height))
        pred_camera_pitch  = pred[0, -1]
        pred_camera_height = pred[0, -2]
        print('pred_camera_pitch: {}'.format(pred_camera_pitch))
        print('pred_camera_height: {}'.format(pred_camera_height))
        gt_camera_pitch = pred_camera_pitch
        gt_camera_height = pred_camera_height

        img_h, img_w, _ = img.shape
        img_canvas = deepcopy(img)
        ipm_canvas = deepcopy(img)

        K = self.K
        aug_mat = np.identity(3, dtype=np.float)
        # print(gt_camera_pitch)
        # print(gt_camera_height)
        # print(K)
        H_g2im = self.homograpthy_g2im(gt_camera_pitch, gt_camera_height, K)
        # print('H_crop_ipm: {}'.format(self.H_crop_ipm))
        # print('H_g2im: {}'.format(H_g2im))
        # print('H_ipm2g: {}'.format(self.H_ipm2g))
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop_ipm, np.matmul(H_g2im, self.H_ipm2g)))
        # print('H_im2ipm: {}'.format(H_im2ipm))
        H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))

        P_g2im = self.projection_g2im(gt_camera_pitch, gt_camera_height, self.K)  # used for x=PX (3D to 2D)
        # H_g2im = self.homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2g = np.linalg.inv(H_g2im)
        P_g2gflat = np.matmul(H_im2g, P_g2im)
        # print('H_im2ipm: {}'.format(H_im2ipm))
        im_ipm = cv2.warpPerspective(ipm_canvas / 255., H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)
        ipm_laneline = im_ipm.copy()
        H_g2ipm = np.linalg.inv(self.H_ipm2g)
        for i, lane in enumerate(gt_2dflatlabels):
            # lane = lane[3:]  # remove conf, upper and lower positions
            seq_len = len(lane-5) // 8
            xs = lane[3:3+seq_len][gt_2dgflatflags[i] > 0]
            ys = lane[3+seq_len:3+seq_len*2][gt_2dgflatflags[i] > 0]
            ys = ys[xs >= 0].astype(np.int)
            xs = xs[xs >= 0].astype(np.int)
            # for p in zip(xs, ys):
            #     p = (int(p[0] * img_w), int(p[1] * img_h))
            #     img_canvas = cv2.circle(img_canvas, p, 5, color=(0, 0, 255), thickness=-1)
            for p in range(1, ys.shape[0]):
                img_canvas = cv2.line(img_canvas, (xs[p - 1], ys[p - 1]), (xs[p], ys[p]), [0, 0, 1], 2)

            gflatlane = lane[5+seq_len*5:]
            gflatXs = gflatlane[:seq_len][gt_2dgflatflags[i] > 0] * self.gflatXnorm
            gflatYs = gflatlane[seq_len:seq_len*2][gt_2dgflatflags[i] > 0] * self.gflatYnorm
            x_ipm, y_ipm = self.homographic_transformation(H_g2ipm, gflatXs, gflatYs)
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)
            for k in range(1, x_ipm.shape[0]):
                ipm_laneline = cv2.line(ipm_laneline, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), [0, 0, 1], 2)
                # ipm_laneline = cv2.circle(ipm_laneline, (x_ipm[k], y_ipm[k]), 5, color=(255, 0, 0), thickness=-1)
        ipm_laneline = (ipm_laneline * 255).astype(np.uint8)
        # cv2.imshow('fff', ipm_laneline)
        # cv2.waitKey(0)
        # exit()
        if pred is None:
            print('Why')
            return img_canvas, ipm_laneline

        # pred_camera_pitch = pred[0, -1]
        # pred_camera_height = pred[0, -2]
        # print('pred_camera_pitch: {}'.format(pred_camera_pitch))
        # print('pred_camera_height: {}'.format(pred_camera_height))

        # H_g2im = self.homograpthy_g2im(pred_camera_pitch, pred_camera_height, K)
        # H_im2ipm = np.linalg.inv(np.matmul(self.H_crop_ipm, np.matmul(H_g2im, self.H_ipm2g)))
        # H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat))
        #
        # P_g2im = self.projection_g2im(pred_camera_pitch, pred_camera_height, self.K)  # used for x=PX (3D to 2D)
        # # H_g2im = self.homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # H_im2g = np.linalg.inv(H_g2im)
        # P_g2gflat = np.matmul(H_im2g, P_g2im)
        #
        # ipm_canvas = deepcopy(img)
        # im_ipm = cv2.warpPerspective(ipm_canvas / 255., H_im2ipm, (self.ipm_w, self.ipm_h))
        # im_ipm = np.clip(im_ipm, 0, 1)
        #
        # ipm_laneline = im_ipm.copy()
        # H_g2ipm = np.linalg.inv(self.H_ipm2g)

        P_gt = np.matmul(self.H_crop_im, H_g2im)
        P_gt = np.matmul(aug_mat, P_gt)
        pred = pred[pred[:, 1].astype(int) == 1]
        matches, accs, _ = self.get_metrics(pred, idx)
        for i, lane in enumerate(pred):
            lower, upper = lane[4], lane[5]
            zlane = lane[10:14]
            lane = lane[6:10]  # remove upper, lower positions
            ys = np.linspace(lower, upper, num=100)
            xs = np.polyval(lane, ys)
            zs = np.polyval(zlane, ys)
            pred_ys = ys * self.gflatYnorm
            pred_xs = xs * self.gflatXnorm
            pred_zs = zs * self.gflatZnorm
            pred_xs, pred_ys = self.projective_transformation(P_g2gflat, pred_xs, pred_ys, pred_zs)
            valid_indices = np.logical_and(np.logical_and(pred_ys > 0, pred_ys < 200),
                                           np.logical_and(pred_xs > 3 * self.x_min, pred_xs < 3 * self.x_max))
            pred_xs = pred_xs[valid_indices]
            pred_ys = pred_ys[valid_indices]
            if pred_xs.shape[0] < 2 or np.sum(np.logical_and(pred_xs > self.x_min, pred_xs < self.x_max)) < 2:
                continue
            pred_ipm_xs, pred_ipm_ys = self.homographic_transformation(H_g2ipm, pred_xs, pred_ys)
            pred_ipm_xs = pred_ipm_xs.astype(np.int)
            pred_ipm_ys = pred_ipm_ys.astype(np.int)
            for k in range(1, pred_ipm_xs.shape[0]):
                ipm_laneline = cv2.line(ipm_laneline, (pred_ipm_xs[k - 1], pred_ipm_ys[k - 1]),
                                        (pred_ipm_xs[k], pred_ipm_ys[k]),
                                        [255, 0, 0], 2)

            pred_x2d, pred_y2d = self.homographic_transformation(P_gt, pred_xs, pred_ys)
            pred_x2d = (pred_x2d * self.w_net / self.w_org).astype(np.int)
            pred_y2d = (pred_y2d * self.h_net / self.h_org).astype(np.int)
            for k in range(1, pred_x2d.shape[0]):
                img_canvas = cv2.line(img_canvas, (pred_x2d[k - 1], pred_y2d[k - 1]), (pred_x2d[k], pred_y2d[k]),
                                      [255, 0, 0], 2)

        return img_canvas, ipm_laneline

    def draw_3dannotation(self, idx, pred=None, img=None, cls_pred=None):
        # _, _, draw_gt_xsys, draw_gt_zsys, draw_gt_flags, \
        # draw_gt_camera_pitch, draw_gt_camera_height, draw_gtground_3dlanes, _ = self.__getitem__(idx)
        _, gt_2dflatlabels, gt_2dgflatflags, gt_camera_pitch, gt_camera_height, _ = \
            self.__getitem__(idx, transform=False)

        img, ipm_img = img
        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax1.imshow(img)

        ax2 = fig.add_subplot(232)
        ax2.imshow(ipm_img)

        ax = fig.add_subplot(233, projection='3d')
        for i in range(gt_2dflatlabels.shape[0]):
            lane = gt_2dflatlabels[i]
            seq_len = len(lane-5) // 8
            lane3D = lane[5+2*seq_len:5+5*seq_len]
            Xs = lane3D[:seq_len][gt_2dgflatflags[i] > 0] * self.gflatXnorm
            Ys = lane3D[seq_len:seq_len * 2][gt_2dgflatflags[i] > 0] * self.gflatYnorm
            Zs = lane3D[seq_len * 2:seq_len * 3][gt_2dgflatflags[i] > 0] * self.gflatYnorm
            ax.plot(Xs, Ys, Zs, color=[0, 0, 1])

        if pred is None:
            ax.set_xlabel('x axis')
            ax.set_ylabel('y axis')
            ax.set_zlabel('z axis')
            bottom, top = ax.get_zlim()
            ax.set_xlim(-20, 20)
            ax.set_ylim(0, 100)
            ax.set_zlim(min(bottom, -1), max(top, 1))
            plt.show()
            print('why')
            return plt
        pred = pred[pred[:, 1].astype(int) == 1]
        matches, accs, _ = self.get_metrics(pred, idx)
        for i, lane in enumerate(pred):
            # print(lane)
            # lane = lane[1:]  # remove conf
            lower, upper = lane[4], lane[5]
            zlane = lane[10:14]
            lane = lane[6:10]  # remove upper, lower positions
            ys = np.linspace(lower, upper, num=100)
            xs = np.polyval(lane, ys)
            zs = np.polyval(zlane, ys)
            pred_ys = ys * self.gflatYnorm
            pred_xs = xs * self.gflatXnorm
            pred_zs = zs * self.gflatZnorm
            ax.plot(pred_xs, pred_ys, pred_zs, color=[1, 0, 0])
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        bottom, top = ax.get_zlim()
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 100)
        ax.set_zlim(min(bottom, -1), max(top, 1))
        # ax.set_zlim(-0.1, 0.1)
        # ax.set_zlim(bottom, top)
        return plt

    def get_metrics(self, lanes, idx):
        # Placeholders
        return [1] * len(lanes), [1] * len(lanes), None

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def homography_crop_resize(self, org_img_size, crop_y, resize_img_size):
        """
            compute the homography matrix transform original image to cropped and resized image
        :param org_img_size: [org_h, org_w]
        :param crop_y:
        :param resize_img_size: [resize_h, resize_w]
        :return:
        """
        # transform original image region to network input region
        ratio_x = resize_img_size[1] / org_img_size[1]
        ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
        H_c = np.array([[ratio_x,          0,                 0],
                        [0,          ratio_y, -ratio_y * crop_y],
                        [0,                0,                 1]])
        return H_c

    def projection_g2im(self, cam_pitch, cam_height, K):
        P_g2c = np.array([[1,                             0,                              0,          0],
                          [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                          [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
        P_g2im = np.matmul(K, P_g2c)
        return P_g2im

    def homograpthy_g2im(self, cam_pitch, cam_height, K):
        # transform top-view region to original image region
        R_g2c = np.array([[1,                             0,                              0],
                          [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                          [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
        H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
        return H_g2im

    def prune_3d_lane_by_visibility(self, lane_3d, visibility):
        lane_3d = lane_3d[visibility > 0, ...]
        return lane_3d

    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max):
        # TODO: solve hard coded range later
        # remove points with y out of range
        # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
        # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

        # remove lane points out of x range
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                         lane_3d[:, 0] < x_max), ...]
        return lane_3d

    def convert_lanes_3d_to_gflat(self, lanes, P_g2gflat):
        """
            Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
            flat ground coordinates [x_gflat, y_gflat, Z]
        :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
        :param P_g2gflat: projection matrix from 3D ground coordinates to frat ground coordinates
        :return:
        """
        # TODO: this function can be simplified with the derived formula
        for lane in lanes:
            # convert gt label to anchor label
            lane_gflat_x, lane_gflat_y = self.projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
            lane[:, 0] = lane_gflat_x
            lane[:, 1] = lane_gflat_y

    def projective_transformation(self, Matrix, x, y, z):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x4 projection matrix
                x (array): original x coordinates
                y (array): original y coordinates
                z (array): original z coordinates
        """
        ones = np.ones((1, len(z)))
        coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

    def homographic_transformation(self, Matrix, x, y):
        """
        Helper function to transform coordinates defined by transformation matrix

        Args:
                Matrix (multi dim - array): 3x3 homography matrix
                x (array): original x coordinates
                y (array): original y coordinates
        """
        ones = np.ones((1, len(y)))
        coordinates = np.vstack((x, y, ones))
        trans = np.matmul(Matrix, coordinates)

        x_vals = trans[0, :] / trans[2, :]
        y_vals = trans[1, :] / trans[2, :]
        return x_vals, y_vals

    def transform_lane_gflat2g(self, h_cam, X_gflat, Y_gflat, Z_g):
        """
            Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
            with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

        :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
        :param X_gflat: X coordinates in flat ground space
        :param Y_gflat: Y coordinates in flat ground space
        :param Z_g: Z coordinates in real 3D ground space
        :return:
        """

        X_g = X_gflat - X_gflat * Z_g / h_cam
        Y_g = Y_gflat - Y_gflat * Z_g / h_cam

        return X_g, Y_g

    def make_lane_y_mono_inc(self, lane):
        """
            Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
            This function trace the y with monotonically increasing y, and output a pruned lane
        :param lane:
        :return:
        """
        idx2del = []
        max_y = lane[0, 1]
        for i in range(1, lane.shape[0]):
            # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
            if lane[i, 1] <= max_y + 3:
                idx2del.append(i)
            else:
                max_y = lane[i, 1]
        lane = np.delete(lane, idx2del, 0)
        return lane

    def resample_laneline_in_y(self, input_lane, y_steps, out_vis=False):
        """
            Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
        :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                           It requires y values of input lane in ascending order
        :param y_steps: a vector of steps in y
        :param out_vis: whether to output visibility indicator which only depends on input y range
        :return:
        """

        # at least two points are included
        assert (input_lane.shape[0] >= 2)

        y_min = np.min(input_lane[:, 1]) - 5
        y_max = np.max(input_lane[:, 1]) + 5

        if input_lane.shape[1] < 3:
            input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

        f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
        f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

        x_values = f_x(y_steps)
        z_values = f_z(y_steps)

        if out_vis:
            output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
            return x_values, z_values, output_visibility
        return x_values, z_values


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)















