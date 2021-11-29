'''
Author: zhonzxad
Date: 2021-11-24 15:29:20
LastEditTime: 2021-11-29 19:58:27
LastEditors: zhonzxad
'''
import argparse
import sys

import torch

sys.path.append("..")

from .loss.bceloss import BCELoss2d
from .loss.celoss import CELOSS, CELoss2d
from .loss.diceloss import Dice_Loss, DiceLoss
from .loss.fscore import FocalLoss
from .loss.iouloss import bbox_overlaps_ciou


def loss_func(output, png, label, this_device="cuda:0"):
    diceloss = DiceLoss()
    celoss   = CELoss2d()
    bceloss  = BCELoss2d()
    floss    = FocalLoss()

    loss = None
    if this_device.type == 'cuda':
        diceloss = diceloss.to(this_device)
        celoss   = celoss.to(this_device)
        # loss = lossF.to(this_device)
    
    dice_loss = diceloss(output, label)
    # bce_loss = bceloss(output, label)
    ce_loss = celoss(output, png)
    # loss = FocalLoss()(output, label)
    # f_scoreloss = floss(output, label)

    loss = ce_loss + dice_loss
    
    # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
    # 如果某一值不需要，将其设置未0
    return loss, ce_loss, torch.zeros_like(loss), dice_loss, torch.zeros_like(loss)
