""" Evaluation script for the CULane metric on the LLAMAS dataset.

This script will compute the F1, precision and recall metrics as described in the CULane benchmark.

The predictions format is the same one used in the CULane benchmark.
In summary, for every annotation file:
    labels/a/b/c.json
There should be a prediction file:
    predictions/a/b/c.lines.txt
Inside each .lines.txt file each line will contain a sequence of points (x, y) separated by spaces.
For more information, please see https://xingangpan.github.io/projects/CULane.html

This script uses two methods to compute the IoU: one using an image to draw the lanes (named `discrete` here) and
another one that uses shapes with the shapely library (named `continuous` here). The results achieved with the first
method are very close to the official CULane implementation. Although the second should be a more exact method and is
faster to compute, it deviates more from the official implementation. By default, the method closer to the official
metric is used.
"""

import os
import argparse
from functools import partial

import cv2
import numpy as np
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from scipy.interpolate import interp1d


FLCP_IMG_RES = (720, 1280)

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    # y_min = np.min(input_lane[:, 1])-5
    y_min = np.min(input_lane[:, 1])
    # y_max = np.max(input_lane[:, 1])+5
    y_max = np.max(input_lane[:, 1])

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

def draw_lane(lane, img=None, img_shape=None, width=30):
    """Draw a lane (a list of points) on an image by drawing a line with width `width` through each
    pair of points i and i+i"""
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1,), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=FLCP_IMG_RES):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys by drawing the lanes on
    an image"""
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # IoU by the definition: sum all intersections (binary and) and divide by the sum of the union (binary or)
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys using the area between each
    pair of points"""
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]
    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area
    return ious

def interpolate_lane(points, n=50):
    """Spline interpolation of a lane. Used on the predictions"""
    x = [x for x, _ in points]
    y = [y for _, y in points]
    # print('x', max(x), min(x), len(x), type(x), x)
    # print('y', max(y), min(y), len(y), type(y), y)
    # assert len(x)==len(y)
    # assert len(x) > 0
    # assert len(y) > 0
    tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
    u = np.linspace(0., 1., n)
    return np.array(splev(u, tck)).T

def culane_metric(pred, anno, width=30, iou_threshold=0.5, unofficial=False, img_shape=FLCP_IMG_RES):
    """Computes CULane's metric for a single image"""
    # print(type(pred))
    # print(type(anno))
    if len(pred) == 0:
        print("this pred has no data")
        return 0, 0, len(anno)
    if len(anno) == 0:
        print("this anno has no data")
        return 0, len(pred), 0
    interp_pred = np.array([interpolate_lane(pred_lane, n=50) for pred_lane in pred])  # (4, 50, 2)
    anno = np.array([np.array(anno_lane) for anno_lane in anno], dtype=object)
    if unofficial:
        ious = continuous_cross_iou(interp_pred, anno, width=width)
    else:
        ious = discrete_cross_iou(interp_pred, anno, width=width, img_shape=img_shape)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    return tp, fp, fn


def load_prediction_list(pred_dir, anno_names):
    # pred_files = os.listdir(pred_dir)
    data = []
    for pred_file in anno_names:
        pred_file = os.path.join(pred_dir, pred_file)
        with open(pred_file, 'r') as pred_obj:
            strlanes = pred_obj.readlines()
        lanes = []
        for strlane in strlanes:
            strpts = strlane.split(' ')
            y_gts = [float(y_) for y_ in strpts[1::2]]
            x_gts = [float(x_) for x_ in strpts[::2]]
            lane = [(x, y) for (x, y) in zip(x_gts, y_gts) if x >= 0]
            # print(lane);exit()
            if len(lane) < 2:
                continue
            lanes.append(lane)
        data.append(lanes)
    return np.array(data, dtype=object)

def load_labels(label_dir):
    anno_files = os.listdir(label_dir)
    data = []
    min_y = 720
    anno_names = []
    for anno_file in anno_files:
        anno_names.append(anno_file)
        anno_file = os.path.join(label_dir, anno_file)
        with open(anno_file, 'r') as anno_obj:
            strlanes = anno_obj.readlines()
        lanes = []
        for strlane in strlanes:
            sbc = int(strlane.split(' ')[-4]) + 1
            strpts = strlane.split(' ')[1:-4]
            y_gts = [float(y_) for y_ in strpts[1::2]]
            x_gts = [float(x_) for x_ in strpts[::2]]
    #         sky_y, road_y = min(y_gts), max(y_gts)
    #         if sky_y < min_y:
    #             min_y = sky_y
    # print('min_y: {}'.format(min_y))
    # for anno_file in anno_files:
    #     anno_file = os.path.join(label_dir, anno_file)
    #     with open(anno_file, 'r') as anno_obj:
    #         strlanes = anno_obj.readlines()
    #     lanes = []
    #     for strlane in strlanes:
    #         sbc = int(strlane.split(' ')[-4]) + 1
    #         strpts = strlane.split(' ')[1:-4]
    #         y_gts = [float(y_) for y_ in strpts[1::2]]
    #         x_gts = [float(x_) for x_ in strpts[::2]]
            lane = [(x, y) for (x, y) in zip(x_gts, y_gts) if x >= 0]
            if len(lane) < 2:
                continue
            # rs_x_gts, _, in_flags = resample_laneline_in_y(np.array(lane), y_steps=np.arange(min_y, 720), out_vis=True)
            # lane = [(x, y) for (x, y) in zip(rs_x_gts[in_flags], np.arange(min_y, 720)[in_flags]) if x >= 0]
            # if len(lane) < 2:
            #     continue
            lanes.append(lane)
        data.append(lanes)
    # exit()
    return np.array(data, dtype=object), anno_names


def eval_predictions(pred_dir, anno_dir, width=30, unofficial=True, sequential=False):
    """Evaluates the predictions in pred_dir and returns CULane's metrics (precision, recall, F1 and its components)"""
    print(f'Loading annotation data ({anno_dir})...')
    annotations, anno_names = load_labels(anno_dir)
    # print('type(annotations): {}'.format(type(annotations)))
    print('len(annotations): {}'.format(len(annotations)))
    print(f'Loading prediction data ({pred_dir})...')
    predictions = load_prediction_list(pred_dir, anno_names)
    # print('type(predictions): {}'.format(type(predictions)))
    print('len(predictions): {}'.format(len(predictions)))
    # predictions = annotations
    # annotations = predictions
    # annotations = predictions
    # anno_num = 0
    # pred_num = 0
    # for i in range(len(annotations)):
    #     anno = annotations[i]
    #     pred = predictions[i]
    #     print('anno')
    #     print(len(anno), type(anno))
    #     for ann in anno:
    #         print(type(ann))
    #         print(len(ann))
    #         # for ap in ann:
    #         #     print(ap)
    #         break
    #     print('pred')
    #     print(len(pred), type(pred))
    #     for pre in pred:
    #         print(type(pre))
    #         print(len(pre))
    #         # for pp in pre:
    #         #     print(pp)
    #         break
    #     exit()
    #     anno_num += len(annotations[i])
    #     pred_num += len(predictions[i])
    # print('anno_num: {}'.format(anno_num))
    # print('pred_num: {}'.format(pred_num))
    # exit()
    print('Calculating metric {}...'.format('sequentially' if sequential else 'in parallel'))
    if sequential:
        results = t_map(partial(culane_metric, width=width, unofficial=unofficial, img_shape=FLCP_IMG_RES),
                        predictions, annotations)
    else:
        results = p_map(partial(culane_metric, width=width, unofficial=unofficial, img_shape=FLCP_IMG_RES),
                        predictions, annotations)
    total_tp = sum(tp for tp, _, _ in results)
    total_fp = sum(fp for _, fp, _ in results)
    total_fn = sum(fn for _, _, fn in results)
    if total_tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric on the LLAMAS dataset")
    parser.add_argument("--pred_dir", help="Path to directory containing the predicted lanes", required=True)
    parser.add_argument("--anno_dir", help="Path to directory containing the annotated lanes", required=True)
    parser.add_argument("--width", type=int, default=30, help="Width of the lane")
    parser.add_argument("--sequential", action='store_true', help="Run sequentially instead of in parallel")
    parser.add_argument("--unofficial", action='store_true', help="Use a faster but unofficial algorithm")

    return parser.parse_args()


def main():
    args = parse_args()
    results = eval_predictions(args.pred_dir,
                               args.anno_dir,
                               width=args.width,
                               unofficial=args.unofficial,
                               sequential=args.sequential)

    header = '=' * 20 + ' Results' + '=' * 20
    print(header)
    for metric, value in results.items():
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print('=' * len(header))


if __name__ == '__main__':
    main()
