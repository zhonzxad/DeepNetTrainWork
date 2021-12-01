'''
Author: zhonzxad
Date: 2021-11-23 10:33:48
LastEditTime: 2021-12-01 10:10:24
LastEditors: zhonzxad
'''

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs                       , axis=[0, 1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    
    return score if not (math.isnan(score) or math.isinf(score)) else torch.zeros(0)

def FScoreLoss(input, target, gamma=2, alpha=0.25):
    n, c, h, w = input.size()
    nt, ht, wt, ct = target.size()

    target = target.permute(0, 3, 1, 2)

    pt = torch.sigmoid(input)
    loss = - alpha * (1 - pt) ** gamma * target * torch.log(pt) - \
                (1 - alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt)
    
    loss = torch.mean(loss)
    
    return loss

def f_score_sklean(inputs, target):
    """
    sklean的方法, 需要将标签转换为列表
    sklearn.metrics.f1_score(y_true, 
                 y_pred,
                 labels=None, 
                 pos_label=1, 
                 average=’binary’, 
                 sample_weight=None)
    average : [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’] 
            多类/多标签目标需要此参数。默认为‘binary’，即二分类
    """
    prob_all = []
    lable_all = []

    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    inputs = inputs.permute(0, 2, 3, 1)  # 全部转换为 [n,h,w,c]

    inputs = inputs.cpu().detach().numpy()  # 转换为numpy
    target = target.cpu().detach().numpy()

    prob_all.extend(np.argmax(inputs, axis=-1)) #求每一行的最大值索引
    lable_all.extend(np.argmax(target, axis=-1))

    f1 = []
    for i in range(n):
        l = f1_score(lable_all[i], prob_all[i], average="binary")
        f1.extend(l)

    return f1_score(lable_all, prob_all, average="binary")

def f_score_1(inputs, target, cls_weights, num_classes=2, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    target = target.long()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


class FocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, cls_weights, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.cls_weights = cls_weights

    def forward(self, input, target):
        
        return f_score_1(input, target, self.cls_weights)
