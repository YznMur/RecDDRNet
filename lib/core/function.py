# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import Map16, Video
# from utils.DenseCRF import DenseCRF
from clearml import Task, Logger

import utils.distributed as dist

VideoCap = Video('./output/cdOffice.mp4')
map16 = Map16(VideoCap)


def compute_segmentation_metrics(confusion_matrix):
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    union = pos + res - tp
    valid_iou = union > 0
    valid_acc = pos > 0

    IoU_array = np.full(tp.shape, np.nan, dtype=np.float64)
    mean_acc_array = np.full(tp.shape, np.nan, dtype=np.float64)

    IoU_array[valid_iou] = tp[valid_iou] / union[valid_iou]
    mean_acc_array[valid_acc] = tp[valid_acc] / pos[valid_acc]

    pixel_acc = tp.sum() / pos.sum() if pos.sum() > 0 else 0.0
    mean_acc = np.nanmean(mean_acc_array) if np.any(valid_acc) else 0.0
    mean_IoU = np.nanmean(IoU_array) if np.any(valid_iou) else 0.0

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _, acc = model(images, labels)
        loss = losses.mean()
        acc  = acc.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        ave_acc.update(acc.item())

        if config.TRAIN.USE_POLY_LR:
            lr = adjust_learning_rate(optimizer,
                                    base_lr,
                                    num_iters,
                                    i_iter+cur_iters)
        else:
            lr = base_lr

        if dist.get_rank() == 0:
            Logger.current_logger().report_scalar(
                title="Learning Rate",
                series="Poly LR",
                value=lr,
                iteration=global_steps
            )
        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    eval_index = min(config.TEST.OUTPUT_INDEX, nums - 1) if nums > 1 else 0
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            image = image.cuda()
            label = label.long().cuda()

            losses, pred, _ = model(image, label)
            if label.dim() == 4:
                b, t, h, w = label.size()
                label = label.view(b * t, h, w)
            size = label.size()
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    selected_mean_IoU = 0.0
    selected_IoU_array = None
    for i in range(nums):
        mean_IoU, IoU_array, _, _ = compute_segmentation_metrics(
            confusion_matrix[..., i]
        )
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
        if i == eval_index:
            selected_mean_IoU = mean_IoU
            selected_IoU_array = IoU_array

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', selected_mean_IoU, global_steps)
    for class_idx, class_iou in enumerate(selected_IoU_array):
        writer.add_scalar(f'valid_class_iou/class_{class_idx}', class_iou, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), selected_mean_IoU, selected_IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            # print(batch,len(batch))
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            # # crf used for post-processing
            # postprocessor = DenseCRF(   )
            # # image
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]
            # timage = image.squeeze(0)
            # timage = timage.numpy().copy().transpose((1,2,0))
            # timage *= std
            # timage += mean
            # timage *= 255.0
            # timage = timage.astype(np.uint8)
            # # pred
            # tprob = torch.softmax(pred, dim=1)[0].cpu().numpy()
            # pred = postprocessor(np.array(timage, dtype=np.uint8), tprob)    
            # pred = torch.from_numpy(pred).unsqueeze(0)
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(image, pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mean_IoU, IoU_array, _, _ = compute_segmentation_metrics(
                    confusion_matrix
                )
                logging.info('mIoU: %.4f' % (mean_IoU))

    mean_IoU, IoU_array, pixel_acc, mean_acc = compute_segmentation_metrics(
        confusion_matrix
    )

    return mean_IoU, IoU_array, pixel_acc, mean_acc


import cv2
def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    
    # Desired output size for the overlay (width, height)
    TARGET_WIDTH = 1024
    TARGET_HEIGHT = 512
    
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            if len(batch) == 3:
                image, size, name = batch
                label = None
            elif len(batch) == 4:
                image, label, size, name = batch

            size = size[0]   # original image size (h, w)

            # Get prediction and resize to original image size (as before)
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                h, w = int(size[0]), int(size[1])
                pred = F.interpolate(
                    pred, size=(h, w),
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                # Reconstruct original image (RGB)
                img = image.squeeze(0).numpy().transpose((1, 2, 0))
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = (img * 255.0).clip(0, 255).astype(np.uint8)

                # Get class prediction
                _, pred = torch.max(pred, dim=1)
                pred = pred.squeeze(0).cpu().numpy()   # shape: (H, W)

                # === CHANGE: Resize BOTH image and prediction to target size BEFORE overlay ===
                img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
                pred_resized = cv2.resize(pred.astype(np.uint8), (TARGET_WIDTH, TARGET_HEIGHT), 
                                          interpolation=cv2.INTER_NEAREST)

                # Now create the overlay at exactly 1024x512
                map16.visualize_result(img_resized, pred_resized, sv_dir, name[0] + '.jpg')

        # VideoCap.releaseCap()   # keep if you need it
