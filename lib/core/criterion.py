# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])

class FocalLoss(nn.Module):
    def __init__(self, ignore_label=-1, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            score = F.interpolate(
                input=score,
                size=(h, w),
                mode='bilinear',
                align_corners=config.MODEL.ALIGN_CORNERS
            )

        logpt = F.log_softmax(score, dim=1)
        pt = torch.exp(logpt)

        #  fix ignore_label BEFORE gather
        target_clone = target.clone()
        target_clone[target_clone == self.ignore_label] = 0

        logpt = logpt.gather(1, target_clone.unsqueeze(1))  # [B,1,H,W]
        pt = pt.gather(1, target_clone.unsqueeze(1))        # [B,1,H,W]

        # remove channel dim → [B,H,W]
        logpt = logpt.squeeze(1)
        pt = pt.squeeze(1)

        mask = target != self.ignore_label

        logpt = logpt[mask]
        pt = pt[mask]

        if self.alpha is not None:
            at = self.alpha[target[mask]]
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])



class CombinedLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None,
                 gamma=2.0,
                 ce_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()

        self.ce = CrossEntropy(ignore_label=ignore_label, weight=weight)
        self.focal = FocalLoss(
            ignore_label=ignore_label,
            gamma=gamma,
            alpha=weight   
        )

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, score, target):
        ce_loss = self.ce(score, target)
        focal_loss = self.focal(score, target)

        return self.ce_weight * ce_loss + self.focal_weight * focal_loss