# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, validate
from core.criterion import OhemCrossEntropy, CrossEntropy
from utils.utils import create_logger, FullModel


SPLIT_MAP = {
    'train': 'TRAIN_SET',
    'val': 'TEST_SET',
    'test': 'TEST_SET',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/rsm/ddrnet23.yaml",
                        type=str)
    parser.add_argument('--split',
                        help='dataset split to evaluate: train, val, or test',
                        default='val',
                        choices=['train', 'val', 'test'],
                        type=str)
    parser.add_argument('--model-file',
                        help='path to model checkpoint (overrides config.TEST.MODEL_FILE)',
                        default=None,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def build_model(config, model_state_file, logger):
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    logger.info('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    return model


def build_dataset(config, list_path, split, logger):
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=list_path,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)
    logger.info('=> dataset: {} samples from {}'.format(len(dataset), list_path))
    return dataset


def evaluate_single_frame(config, model, dataset, logger):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(
        config, dataset, loader, model, sv_pred=False)

    class_names = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]
    logger.info('=' * 60)
    logger.info('Single-frame evaluation results:')
    logger.info('  Mean IoU:      {:.4f}'.format(mean_IoU))
    logger.info('  Pixel Accuracy: {:.4f}'.format(pixel_acc))
    logger.info('  Mean Accuracy:  {:.4f}'.format(mean_acc))
    logger.info('  Per-class IoU:')
    for idx, iou in enumerate(IoU_array):
        name = class_names[idx] if idx < len(class_names) else f'class_{idx}'
        logger.info('    {:2d} {:<18s} IoU: {:.4f}'.format(
            idx, name, iou if not np.isnan(iou) else 0.0))
    logger.info('=' * 60)

    end = timeit.default_timer()
    logger.info('Time: {:.1f}s'.format(end - start))
    return mean_IoU, IoU_array, pixel_acc, mean_acc


class _NoOpWriter:
    def add_scalar(self, *a, **kw): pass
    def add_scalars(self, *a, **kw): pass


def evaluate_sequence(config, model, dataset, logger):
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=config.LOSS.OHEMTHRES,
            min_kept=config.LOSS.OHEMKEEP,
            weight=None)
    else:
        criterion = CrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            weight=None)

    full_model = FullModel(model, criterion)
    full_model = full_model.cuda()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    writer_dict = {
        'writer': _NoOpWriter(),
        'valid_global_steps': 0,
    }

    start = timeit.default_timer()
    valid_loss, mean_IoU, IoU_array = validate(
        config, loader, full_model, writer_dict)

    class_names = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]
    logger.info('=' * 60)
    logger.info('Sequence evaluation results:')
    logger.info('  Valid Loss:     {:.4f}'.format(valid_loss))
    logger.info('  Mean IoU:       {:.4f}'.format(mean_IoU))
    logger.info('  Per-class IoU:')
    for idx, iou in enumerate(IoU_array):
        name = class_names[idx] if idx < len(class_names) else f'class_{idx}'
        logger.info('    {:2d} {:<18s} IoU: {:.4f}'.format(
            idx, name, iou if not np.isnan(iou) else 0.0))
    logger.info('=' * 60)

    end = timeit.default_timer()
    logger.info('Time: {:.1f}s'.format(end - start))
    return mean_IoU, IoU_array


def main():
    args = parse_args()
    split = args.split

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'eval_{}'.format(split))

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    if args.model_file:
        model_state_file = args.model_file
    elif config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')

    model = build_model(config, model_state_file, logger)

    list_path = config.DATASET[SPLIT_MAP[split]]
    dataset = build_dataset(config, list_path, split, logger)

    logger.info('Evaluating split: {} (list: {})'.format(split, list_path))
    is_sequence = getattr(dataset, 'prebuilt_sequence', False)
    logger.info('Sequence data: {}'.format(is_sequence))

    if is_sequence:
        evaluate_sequence(config, model, dataset, logger)
    else:
        evaluate_single_frame(config, model, dataset, logger)


if __name__ == '__main__':
    main()
