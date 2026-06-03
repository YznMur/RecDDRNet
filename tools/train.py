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
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy, CombinedLoss
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

from clearml import Task, Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet_slim.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def get_output_paths(cfg, cfg_name):
    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_stem = os.path.basename(cfg_name).split('.')[0]
    final_output_dir = os.path.join(cfg.OUTPUT_DIR, dataset, cfg_stem)
    tensorboard_log_dir = os.path.join(cfg.LOG_DIR, dataset, model)
    return final_output_dir, tensorboard_log_dir


def main():
    args = parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank < 0 and env_local_rank >= 0:
        args.local_rank = env_local_rank

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1 or args.local_rank >= 0
    is_main_process = (not distributed) or args.local_rank in (-1, 0)

    task = None
    clearml_logger = None
    if is_main_process:
        task = Task.init(
            project_name="DDRNet_Segmentation",
            task_name=f"train_{config.MODEL.NAME}_{config.DATASET.DATASET}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            auto_connect_frameworks={
                "pytorch": True,
                "tensorboard": True
            }
        )

        task.connect({
            "args": vars(args),
            "config": config
        })

        clearml_logger = Logger.current_logger()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    if distributed:
        if args.local_rank < 0:
            raise ValueError("Distributed launch detected but LOCAL_RANK was not set.")
        print("---------------devices:", args.local_rank)
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    else:
        device = torch.device('cuda:{}'.format(gpus[0] if gpus else 0))

    if is_main_process:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'train')
        writer = SummaryWriter(tb_log_dir)
        logger.info(pprint.pformat(args))
        logger.info(config)
    else:
        final_output_dir, _ = get_output_paths(config, args.cfg)
        os.makedirs(final_output_dir, exist_ok=True)
        logger = logging.getLogger(f"train_rank_{args.local_rank}")
        writer = NullWriter()

    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if distributed:
        torch.distributed.barrier()

    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d

    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')

        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)

        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        torch.distributed.barrier()

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    train_dataset_kwargs = dict(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)
    if config.DATASET.DATASET == 'rsm':
        train_dataset_kwargs['sequence'] = config.TRAIN.SEQUENCE_MODE
        train_dataset_kwargs['sequence_len'] = config.TRAIN.SEQUENCE_LEN

    train_dataset = eval('datasets.'+config.DATASET.DATASET)(**train_dataset_kwargs)

    train_sampler = get_sampler(train_dataset)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    extra_epoch_iters = 0

    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset_kwargs = dict(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)
        if config.DATASET.DATASET == 'rsm':
            extra_train_dataset_kwargs['sequence'] = config.TRAIN.SEQUENCE_MODE
            extra_train_dataset_kwargs['sequence_len'] = config.TRAIN.SEQUENCE_LEN

        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(**extra_train_dataset_kwargs)

        extra_train_sampler = get_sampler(extra_train_dataset)

        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)

        extra_epoch_iters = np.int(extra_train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    test_dataset_kwargs = dict(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    test_dataset = eval('datasets.'+config.DATASET.DATASET)(**test_dataset_kwargs)

    test_sampler = get_sampler(test_dataset)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    elif config.LOSS.USE_CombinedLoss:
        criterion = CombinedLoss(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            weight=train_dataset.class_weights,
            gamma=2.0,
            ce_weight=1.0,
            focal_weight=1.0
        )

    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)
    model = FullModel(model, criterion)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())

        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(
                                params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    checkpoint_save_interval = 50

    model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')

    if os.path.isfile(model_state_file):

        logger.info("=> Resuming automatically from checkpoint")

        checkpoint = torch.load(model_state_file, map_location='cpu')

        best_mIoU = checkpoint['best_mIoU']
        last_epoch = checkpoint['epoch']

        model.module.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("=> Loaded checkpoint (epoch {})".format(last_epoch))

        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()

    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    for epoch in range(last_epoch, end_epoch):
        if clearml_logger is not None:
            clearml_logger.report_scalar(
                "Train",
                "Epoch",
                epoch,
                iteration=epoch
            )
        current_trainloader = extra_trainloader if epoch >= config.TRAIN.END_EPOCH else trainloader

        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        if epoch >= config.TRAIN.END_EPOCH:

            train(config, epoch-config.TRAIN.END_EPOCH,
                  config.TRAIN.EXTRA_EPOCH,
                  extra_epoch_iters,
                  config.TRAIN.EXTRA_LR,
                  extra_iters,
                  extra_trainloader,
                  optimizer,
                  model,
                  writer_dict)

        else:

            train(config, epoch,
                  config.TRAIN.END_EPOCH,
                  epoch_iters,
                  config.TRAIN.LR,
                  num_iters,
                  trainloader,
                  optimizer,
                  model,
                  writer_dict)

        if epoch % 10 == 0:

            valid_loss, mean_IoU, IoU_array = validate(
                config,
                testloader,
                model,
                writer_dict
            )

            if clearml_logger is not None:
                clearml_logger.report_scalar(
                    "Validation", "Mean IoU", mean_IoU, epoch
                )

                clearml_logger.report_scalar(
                    "Validation", "Loss", valid_loss, epoch
                )

                for class_idx, class_iou in enumerate(IoU_array):
                    clearml_logger.report_scalar(
                        "Validation Class IoU",
                        f"Class {class_idx}",
                        float(class_iou),
                        epoch
                    )

            if args.local_rank <= 0:

                if mean_IoU > best_mIoU:

                    best_mIoU = mean_IoU

                    best_model_path = os.path.join(final_output_dir, 'best.pth')

                    torch.save(model.module.state_dict(), best_model_path)

                    if task is not None:
                        task.update_output_model(
                            model_path=best_model_path,
                            name="DDRNet_best_model"
                        )

                msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                            valid_loss, mean_IoU, best_mIoU)

                logging.info(msg)
                logging.info(IoU_array)

        if args.local_rank <= 0:

            logger.info('=> saving checkpoint')

            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            if (epoch + 1) % checkpoint_save_interval == 0:
                epoch_weights_path = os.path.join(
                    final_output_dir, f'epoch_{epoch + 1:04d}.pth'
                )
                epoch_checkpoint_path = os.path.join(
                    final_output_dir, f'checkpoint_epoch_{epoch + 1:04d}.pth.tar'
                )

                logger.info('=> saving periodic weights to %s', epoch_weights_path)
                torch.save(model.module.state_dict(), epoch_weights_path)
                torch.save({
                    'epoch': epoch+1,
                    'best_mIoU': best_mIoU,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, epoch_checkpoint_path)

    if args.local_rank <= 0:

        final_model_path = os.path.join(final_output_dir, 'final_state.pth')

        torch.save(model.module.state_dict(), final_model_path)

        if task is not None:
            task.update_output_model(
                model_path=final_model_path,
                name="DDRNet_final_model"
            )

        writer_dict['writer'].close()

        end = timeit.default_timer()

        logger.info('Hours: %d' % np.int((end-start)/3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
