# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def pixel_acc(self, pred, label):
    if isinstance(pred, (list, tuple)):
        output_index = min(config.TEST.OUTPUT_INDEX, len(pred) - 1)
        pred = pred[output_index]
    if pred.shape[2] != label.shape[1] or pred.shape[3] != label.shape[2]:
        pred = F.interpolate(pred, size=label.shape[1:], mode="bilinear", align_corners=False)
    _, preds = torch.max(pred, dim=1)
    ignore_label = getattr(self.loss, 'ignore_label', -1)
    valid = (label != ignore_label).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, *args, **kwargs):
    outputs = self.model(inputs, *args, **kwargs)
    loss = self.loss(outputs, labels)
    acc  = self.pixel_acc(outputs, labels)
    return torch.unsqueeze(loss,0), outputs, acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

import cv2
from PIL import Image

def colorEncode(labelmap, colors, mode='RGB'):
    """
    Encode a label map into a color image.

    Args:
        labelmap (np.ndarray): HxW array of class indices
        colors (np.ndarray): Nx3 color palette
        mode (str): 'RGB' or 'BGR'

    Returns:
        np.ndarray: HxWx3 colored image
    """
    labelmap = labelmap.astype(np.int32)

    h, w = labelmap.shape
    labelmap_rgb = np.zeros((h, w, 3), dtype=np.int32)

    for label in np.unique(labelmap):
        if label < 0 or label >= len(colors):
            continue

        mask = (labelmap == label)
        labelmap_rgb[mask] = colors[label]

    labelmap_rgb = np.clip(labelmap_rgb, 0, 255).astype(np.uint8)

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]

    return labelmap_rgb

class Video(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 480))

    def addImage(self, img, colorMask):
        img = img[:,:,::-1]
        colorMask = colorMask[:,:,::-1]         # shape:
        img = np.concatenate([img, colorMask], axis=1)
        self.cap.write(img)

    def releaseCap(self):
        self.cap.release()


class Map16(object):
    def __init__(self, VideoCap, visualpoint=False):
        self.names = (
            "Field",
            "Grass",
            "Windrow",
            "Unlabeled objects",
            "Obstacles",
            "door",
            "table,desk,coffee",
            "chair,armchair,sofa,bench,swivel,stool",
            "rug",
            "railing",
            "column",
            "refrigerator",
            "stairs,stairway,step",
            "escalator",
            "wall",
            "dog",
            "plant",
            "er1",
            "er2",
            "er3"
        )

        self.colors = np.array([
            [0, 255, 0],     # green
            [255, 165, 0],   # orange 
            [0, 0, 255],     # blue
            [0, 0, 0],       # black
            [255, 0, 0],   # Red
            [140,86,75],
            [227,119,194],
            [127,127,127],
            [188,189,34],
            [23,190,207],
            [255,180,25],
            [0,128,128],
            [50,50,255],   # changed
            [255,0,255],
            [200,0,0],     # changed
            [0,200,0],     # changed
            [255,255,0],
            [0,255,255],
            [128,0,255],
            [255,128,0]
        ], dtype=np.uint8)

        self.outDir = "output/map16"
        self.VideoCap = VideoCap
        self.visualpoint = visualpoint
        
    def visualize_result(self, data, pred, dir, img_name=None):
        img = data.copy()

        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)

        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx]]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        # optional keypoints
        if self.visualpoint:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = np.float32(img_gray)
            corners = cv2.goodFeaturesToTrack(img_gray, 400, 0.01, 10)
            if corners is not None:
                corners = np.int0(corners)
                for i in corners:
                    x, y = i.flatten()
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        if pred_color.shape != img.shape:
            pred_color = cv2.resize(
                pred_color,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        im_vis = (img * 0.6 + pred_color * 0.4).astype(np.uint8)
        # im_vis = pred_color
        # -------------------------------
        # LEGEND (FIRST 5 CLASSES ONLY)
        # -------------------------------
        x0, y0 = 10, 10
        box_h = 22

        for i in range(min(5, len(self.names))):
            color = self.colors[i].tolist()
            name = self.names[i]

            y = y0 + i * box_h

            # color box
            cv2.rectangle(im_vis, (x0, y), (x0 + 15, y + 15), color, -1)

            # text label
            cv2.putText(
                im_vis,
                name,
                (x0 + 20, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # save video frame
        self.VideoCap.addImage(im_vis, pred_color)

        if not os.path.exists(dir):
            os.makedirs(dir)

        Image.fromarray(im_vis).save(
            os.path.join(dir, img_name)
        )


def speed_test(model, size=896, iteration=100):
    input_t = torch.Tensor(1, 3, size, size).cuda()
    feed_dict = {}
    feed_dict['img_data'] = input_t

    print("start warm up")

    for i in range(10):
        model(feed_dict, segSize=(size, size))

    print("warm up done")
    start_ts = time.time()
    for i in range(iteration):
        model(feed_dict, segSize=(size, size))

    torch.cuda.synchronize()
    end_ts = time.time()

    t_cnt = end_ts - start_ts
    print("=======================================")
    print("FPS: %f" % (100 / t_cnt))
    print(f"Inference time {t_cnt/100*1000} ms")
