'''
Author: zhonzxad
Date: 2021-11-24 15:29:20
LastEditTime: 2021-11-25 15:26:14
LastEditors: zhonzxad
'''
import argparse
import sys

sys.path.append("..")

from .loss.bceloss import BCELoss2d
from .loss.celoss import CELOSS, CELoss2d
from .loss.diceloss import Dice_Loss, DiceLoss
from .loss.fscore import FocalLoss
from .loss.iouloss import bbox_overlaps_ciou


def loss_func(output, png, label, this_device="cuda:0"):
    diceloss = DiceLoss()
    celoss   = CELoss2d()

    loss = None
    if this_device.type == 'cuda':
        diceloss = diceloss.to(this_device)
        celoss   = celoss.to(this_device)
        # loss = lossF.to(this_device)
    
    # dice_loss = diceloss(output, label)
    ce_loss = celoss(output, png)
    # loss = FocalLoss()(output, label)

    loss = ce_loss
    
    return loss


