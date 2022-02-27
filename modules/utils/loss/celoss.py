'''
Author: zhonzxad
Date: 2021-10-25 13:18:03
LastEditTime: 2021-11-30 11:54:19
LastEditors: zhonzxad
'''

from warnings import catch_warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def AchieveCE_1(inputs, target):
    target = target.long()
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

def AchieveCE_2(inputs, target):
    target = target.long()

    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    inputs = torch.softmax(inputs, dim=1)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) # [n*h*w, c]
    temp_target = target.view(-1).long()  # [n*h*w]
    # print("\n temp_inputs shape is {} || temp_target shape is {}".format(temp_inputs.shape, temp_target.shape))
    # 以下两个float操作的作用是在amp作用下，防止出现某些精度为nan的情况
    # temp_inputs = temp_inputs.float()
    # temp_target = temp_target.float()
    celoss  = nn.NLLLoss()(F.log_softmax(temp_inputs, dim=-1), temp_target)

    return celoss

def AchieveCE_3(pred, target, num_classes=2):
    n, c, h, w = pred.size()
    nt, ht, wt = target.size()

    # inputs = torch.softmax(inputs, dim=1)
    pred = torch.sigmoid(pred)

    pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    target = target.contiguous().view(-1).long()   # 全部转换为[n*h*w]形式

    CE_loss = nn.CrossEntropyLoss()(pred, target)

    return CE_loss

def AchieveCE_4(pred, target):
    n, c, h, w = pred.size()
    nt, ht, wt, ct = target.size()

    pred = pred.permute(0, 2, 3, 1).view(n, -1, c)
    pred = torch.softmax(pred, dim=-1)
    target = target.view(nt, -1, ct)

    return torch.nn.CrossEntropyLoss()(pred, target)

def CELOSS(pred, targets):
    # pred = torch.sigmoid(pred)
    # targets = targets.long()
    # return nn.CrossEntropyLoss()(pred, targets)
    chanel = pred.shape[1]
    pred = torch.softmax(pred, dim=1)
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
        pass

    # 针对celoss或者bceloss的情况，在使用amp计算时容易出现nan的情况，需要强制将
    # half精度转回float32, 也就是x=x.float()
    # 对于二分类问题，sigmoid等价于softmax

    def forward(self, pred, target):

        return AchieveCE_3(pred, target)

        

