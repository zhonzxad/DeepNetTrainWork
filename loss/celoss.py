'''
Author: zhonzxad
Date: 2021-10-25 13:18:03
LastEditTime: 2021-11-22 19:31:24
LastEditors: zhonzxad
'''

from warnings import catch_warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def AchieveCE_1(inputs, target):
    try:
        n, c, h, w = inputs.size()
        nt, ht, wt = target.size()
        # 如果输出结果与原始结果size不同 双线性插值
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    except AttributeError:
        print("网络输出结果或者标签没有对应的size属性")
    except ValueError:
        print("网络输出结果或者标签的size属性不具备四个属性")

    CE_loss = nn.CrossEntropyLoss()(inputs, target)

    return CE_loss

def AchieveCE_2(inputs, target, num_classes=1):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    # print("\n temp_inputs shape is {} || temp_target shape is {}".format(temp_inputs.shape, temp_target.shape))
    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)

    return CE_loss

def CELOSS(pred, targets):
    # pred = torch.sigmoid(pred)
    # targets = targets.long()
    # return nn.CrossEntropyLoss()(pred, targets)
    chanel = pred.shape[1]
    pred = torch.sigmoid(pred)
    pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, chanel)
    targets = targets.long()
    targets = targets.view(-1)
    return nn.CrossEntropyLoss()(pred, targets)

def Imgndim(img):
    if img.ndim == 2:       # 2维度表示长宽
        return 1            # 单通道(grayscale)
    elif img.ndim == 3:     # 三通道
        assert (img.shape[-1] == 3) # 第三维度表示通道，应为3
        return 3
    else:                   # 异常维度，不是图片了
        return -1

class CELoss2d(nn.Module):
    def __init__(self):
          super(CELoss2d, self).__init__()
 
    def forward(self, pred, targets):
        chanel = pred.shape[1]
        pred = torch.sigmoid(pred)
        pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, chanel)
        targets = targets.long()
        targets = targets.view(-1)
        return nn.CrossEntropyLoss()(pred, targets)

        # 对于二分类问题，sigmoid等价于softmax
        pred = torch.sigmoid(pred)
        ce_loss = AchieveCE_1(pred, targets)
        return ce_loss

