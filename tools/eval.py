import argparse
import json
import os
import pprint
import sys

import logging
import time
import timeit

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

# Allow separate test list via EVAL_TEST_SET config key
SPLIT_TEST_OVERRIDES = {
    'test': 'EVAL_TEST_SET',
}

CLASS_NAMES = ["Field", "Grass", "Windrow", "Unused_objects", "Obstacles"]


class _NoOpWriter:
    def add_scalar(self, *a, **kw): pass
    def add_scalars(self, *a, **kw): pass


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/rsm_ddrnet23slim_keyframe_updater_attn.yaml",
                        type=str)
    parser.add_argument('--splits',
                        help='comma-separated splits to evaluate (train,val,test) or "all"',
                        default='all',
                        type=str)
    parser.add_argument('--model-file',
                        help='path to model checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--save-masks',
                        action='store_true',
                        help='save predicted class masks as PNG files')
    parser.add_argument('--mask-dir',
                        default=None,
                        help='directory for saved masks (default: {output_dir}/masks)')
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
    model = nn.DataParallel(model, device_ids=gpus).cuda(gpus[0])
    return model


def build_dataset(config, list_path, logger):
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    dataset_kwargs = dict(
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
    if config.DATASET.DATASET == 'rsm':
        dataset_kwargs['sequence'] = config.TRAIN.SEQUENCE_MODE
        dataset_kwargs['sequence_len'] = config.TRAIN.SEQUENCE_LEN
    dataset = eval('datasets.'+config.DATASET.DATASET)(**dataset_kwargs)
    logger.info('=> dataset: {} samples from {}'.format(len(dataset), list_path))
    return dataset


def evaluate_single_frame(config, model, dataset, logger, mask_dir=None):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(
        config, dataset, loader, model, sv_pred=False, mask_dir=mask_dir)
    elapsed = timeit.default_timer() - start

    logger.info('  Time: {:.1f}s'.format(elapsed))
    return {
        'mean_iou': float(mean_IoU),
        'pixel_acc': float(pixel_acc),
        'mean_acc': float(mean_acc),
        'iou_per_class': {CLASS_NAMES[i] if i < len(CLASS_NAMES) else 'class_{}'.format(i):
                          float(iou) if not np.isnan(iou) else 0.0
                          for i, iou in enumerate(IoU_array)},
        'time_seconds': elapsed,
    }


def evaluate_sequence(config, model, dataset, logger, mask_dir=None):
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
        config, loader, full_model, writer_dict, mask_dir=mask_dir)
    elapsed = timeit.default_timer() - start

    raw_model = model.module if hasattr(model, 'module') else model
    keyframe_interval = getattr(config.MODEL, 'KEYFRAME_INTERVAL', 1)
    backbone_runs = getattr(raw_model, '_backbone_count', None)
    updater_runs = getattr(raw_model, '_updater_count', None)

    if keyframe_interval > 1:
        logger.info('  Keyframe interval: {}'.format(keyframe_interval))
        logger.info('  Last sequence: backbone={}, updater={}'.format(backbone_runs, updater_runs))

    logger.info('  Time: {:.1f}s ({:.1f}s/sequence)'.format(
        elapsed, elapsed / max(len(dataset), 1)))

    return {
        'valid_loss': float(valid_loss),
        'mean_iou': float(mean_IoU),
        'iou_per_class': {CLASS_NAMES[i] if i < len(CLASS_NAMES) else 'class_{}'.format(i):
                          float(iou) if not np.isnan(iou) else 0.0
                          for i, iou in enumerate(IoU_array)},
        'keyframe_interval': keyframe_interval,
        'backbone_runs': backbone_runs,
        'updater_runs': updater_runs,
        'time_seconds': elapsed,
    }


def run_split(config, model, split, logger, mask_dir=None):
    override_key = SPLIT_TEST_OVERRIDES.get(split)
    if override_key and hasattr(config.DATASET, override_key):
        list_path = getattr(config.DATASET, override_key)
    else:
        list_path = config.DATASET[SPLIT_MAP[split]]
    logger.info('=' * 60)
    logger.info('Evaluating split: {} (list: {})'.format(split, list_path))
    logger.info('=' * 60)

    dataset = build_dataset(config, list_path, logger)
    is_sequence = getattr(dataset, 'prebuilt_sequence', False)
    logger.info('Sequence data: {}'.format(is_sequence))

    if is_sequence:
        results = evaluate_sequence(config, model, dataset, logger, mask_dir=mask_dir)
    else:
        results = evaluate_single_frame(config, model, dataset, logger, mask_dir=mask_dir)

    logger.info('--- {} Results ---'.format(split.upper()))
    if 'valid_loss' in results:
        logger.info('  Loss:     {:.4f}'.format(results['valid_loss']))
    logger.info('  mIoU:     {:.4f}'.format(results['mean_iou']))
    if 'pixel_acc' in results:
        logger.info('  PixelAcc: {:.4f}'.format(results['pixel_acc']))
    for cls_name, cls_iou in results['iou_per_class'].items():
        logger.info('  {:<18s} IoU: {:.4f}'.format(cls_name, cls_iou))

    return results


def save_results(all_results, output_dir, logger):
    metrics_path = os.path.join(output_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Metrics saved to: {}'.format(metrics_path))

    txt_path = os.path.join(output_dir, 'eval_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write('DDRNet Evaluation Results\n')
        f.write('=' * 60 + '\n\n')
        for split, results in all_results.items():
            f.write('Split: {}\n'.format(split.upper()))
            f.write('-' * 40 + '\n')
            if 'error' in results:
                f.write('  ERROR: {}\n\n'.format(results['error']))
                continue
            if 'valid_loss' in results:
                f.write('  Loss:     {:.4f}\n'.format(results['valid_loss']))
            f.write('  mIoU:     {:.4f}\n'.format(results['mean_iou']))
            if 'pixel_acc' in results:
                f.write('  PixelAcc: {:.4f}\n'.format(results['pixel_acc']))
            if 'mean_acc' in results:
                f.write('  MeanAcc:  {:.4f}\n'.format(results['mean_acc']))
            f.write('  Per-class IoU:\n')
            for cls_name, cls_iou in results['iou_per_class'].items():
                f.write('    {:<18s} {:.4f}\n'.format(cls_name, cls_iou))
            if 'keyframe_interval' in results and results['keyframe_interval'] > 1:
                f.write('  Keyframe interval: {}\n'.format(results['keyframe_interval']))
                f.write('  Backbone runs: {}, Updater runs: {}\n'.format(
                    results.get('backbone_runs', 'N/A'),
                    results.get('updater_runs', 'N/A')))
            f.write('  Time: {:.1f}s\n\n'.format(results.get('time_seconds', 0)))
    logger.info('Metrics saved to: {}'.format(txt_path))


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'eval_all')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    if args.model_file:
        model_state_file = args.model_file
    elif hasattr(config.TEST, 'MODEL_FILE') and config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')

    model = build_model(config, model_state_file, logger)

    if args.splits == 'all':
        splits = ['test', 'train', 'val']
    else:
        splits = [s.strip() for s in args.splits.split(',')]

    all_results = {}
    mask_dir = None
    if args.save_masks:
        mask_dir = args.mask_dir or os.path.join(final_output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        logger.info('Masks will be saved to: {}'.format(mask_dir))

    for split in splits:
        try:
            split_mask_dir = os.path.join(mask_dir, split) if mask_dir else None
            all_results[split] = run_split(config, model, split, logger, mask_dir=split_mask_dir)
        except Exception as e:
            logger.error('Error evaluating {}: {}'.format(split, e))
            all_results[split] = {'error': str(e)}

    save_results(all_results, final_output_dir, logger)

    logger.info('')
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    for split, results in all_results.items():
        if 'error' in results:
            logger.info('  {}: ERROR - {}'.format(split.upper(), results['error']))
        else:
            logger.info('  {}: mIoU={:.4f}'.format(split.upper(), results['mean_iou']))
    logger.info('=' * 60)
    logger.info('Done. Results saved to: {}'.format(final_output_dir))


if __name__ == '__main__':
    main()
