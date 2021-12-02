'''
Author: zhonzxad
Date: 2021-11-24 15:29:20
LastEditTime: 2021-12-02 21:34:12
LastEditors: zhonzxad
'''
import torch
# 在文件被引用的初始使用绝对路径
from models.Unit.loss.bceloss import BCELoss2d
from models.Unit.loss.celoss import CELoss2d
from models.Unit.loss.diceloss import DiceLoss
from models.Unit.loss.fscore import FocalLoss
from models.Unit.loss.transferloss import TransferLoss

# from loss import *

def loss_func(output, png, label, cls_weights, this_device="cuda:0"):
    diceloss = DiceLoss()
    celoss   = CELoss2d()
    bceloss  = BCELoss2d()
    floss    = FocalLoss(cls_weights)
    loss = None
    
    if this_device.type == 'cuda':
        diceloss = diceloss.to(this_device)
        celoss   = celoss.to(this_device)
        # loss = lossF.to(this_device)
    
    dice_loss = diceloss(output, label)
    # bce_loss = bceloss(output, label)
    # ce_loss = celoss(output, png)
    focal_loss = floss(output, png)
    # loss = FocalLoss()(output, label)
    

    loss = focal_loss + dice_loss
    
    # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
    # 如果某一值不需要，将其设置未0
    return loss, torch.zeros_like(loss), torch.zeros_like(loss), dice_loss, focal_loss
