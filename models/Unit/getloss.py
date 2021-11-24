'''
Author: zhonzxad
Date: 2021-11-24 15:29:20
LastEditTime: 2021-11-24 15:40:11
LastEditors: zhonzxad
'''
import argparse
import sys

sys.path.append("..")

from loss.bceloss import BCELoss2d
from loss.celoss import CELOSS, CELoss2d
from loss.diceloss import Dice_Loss, DiceLoss
from loss.fscore import FocalLoss
from loss.iouloss import bbox_overlaps_ciou


def loss_func(output, png, label, this_device="cuda:0"):
    lossF = FocalLoss()
    if this_device.type == 'cuda':
        loss = lossF.to(this_device)
    
    loss = lossF(output, label)
    
    return loss
