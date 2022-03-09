'''
Author: zhonzxad
Date: 2021-11-24 15:29:20
LastEditTime: 2021-12-02 21:34:12
LastEditors: zhonzxad
'''
import torch
# 在文件被引用的初始使用绝对路径
from UserProject.modules.utils.loss.celoss import CELoss2d
from UserProject.modules.utils.loss.diceloss import DiceLoss

# from loss import *

"""NOTICE
在使用amp混合精度训练时容易出现nan的一些情况
在使用ce loss 或者 bceloss的时候，会有log的操作，在半精度情况下，一些非常小的数值会被直接舍入到0，log(0)等于啥？——等于nan啊！
"""

def loss_func(output, png, label, cls_weights, this_device):
    diceloss = DiceLoss(weight=cls_weights)
    celoss   = CELoss2d(weight=cls_weights)
    # bceloss  = BCELoss2d()
    # floss    = FocalLoss(cls_weights)
    loss = None
    
    if this_device.type == 'cuda':
        diceloss = diceloss.to(this_device)
        celoss   = celoss.to(this_device)
        # loss = lossF.to(this_device)
    
    dice_loss = diceloss(output, label)
    # bce_loss = bceloss(output, label)
    ce_loss = celoss(output, png)
    # focal_loss = floss(output, png)
    # loss = FocalLoss()(output, label)

    loss = ce_loss + dice_loss
    
    # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
    # 如果某一值不需要，将其设置未0
    return loss, ce_loss, torch.zeros_like(loss), dice_loss, torch.zeros_like(loss)
