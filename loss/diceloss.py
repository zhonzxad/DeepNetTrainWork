import torch
import torch.nn as nn
import torch.nn.functional as F


def AchieveDice_1(inputs, target):
    beta=1
    smooth=1e-5
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    
    return dice_loss


def AchieveDice_2(logits, targets):
    '''
    # https://github.com/pytorch/pytorch/issues/1249
    '''
    class_count = targets.size(0) # 有多少类别
    smooth = 1
    
    probs = torch.sigmoid(logits)
    m1 = probs.view(class_count, -1)
    m2 = targets.view(class_count, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / class_count
    return score

def AchieveDice_3(pred, target):
    '''
    :param y_preds: [bs,num_classes,768,1024]
    :param y_truths: [bs,num_calsses,768,1024]
    :param eps:
    :return:
    '''
    eps=1e-8
    batch_size = pred.size(0)
    num_classes = pred.size(1)
    dices_bs = torch.zeros(batch_size, num_classes)
    for idx in range(batch_size):
        y_pred = pred[idx]    # [num_classes,768,1024]
        y_truth = target      # [num_classes,768,1024]
        intersection = torch.sum(torch.mul(y_pred, y_truth),dim=(1,2)) + eps/2
        union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth), dim=(1, 2)) + eps

        dices_sub = 2 * intersection / union
        dices_bs[idx] = dices_sub

    dices = torch.mean(dices_bs,dim=0)
    dice = torch.mean(dices)
    dice_loss = 1 - dice
    return dice_loss

def Dice_Loss(pred, label):
    '''
    dice计算，是一种集合相似度度量函数，通常用于计算两个样本的相似度，是越大越好。
    比较适用于样本极度不均的情况，一般的情况下，使用 dice loss 会对反向传播造成不利的影响，容易使训练变得不稳定
    pred:网络预测图
    label:标签图
    '''
    dice_loss = AchieveDice_3(pred, label)
    return dice_loss

def AchieveDice_1(input, target, beta=1, smooth=1e-5, num_classes=2):
    b, c, h, w = input.size()
    bt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(input.transpose(1, 2).transpose(2, 3).contiguous().view(b, -1, c), -1)
    temp_target = target.view(bt, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)

    # 最后越小越好
    return dice_loss

def AchieveDice_2(pred, label, eps=1e-8, num_classes=2):
    '''
    :param y_preds: [bs,num_classes,768,1024]
    :param y_truths: [bs,num_calsses,768,1024]
    :param eps:
    :return:
    '''
    pred = torch.sigmoid(pred)
    bs = pred.size(0)
    num_classes = pred.size(1)
    dices_bs = torch.zeros(bs,num_classes)
    for idx in range(bs):
        y_pred  = pred[idx]   # [num_classes,768,1024]
        y_truth = label[idx]   # [num_classes,768,1024]
        intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(1,2)) + eps / 2
        union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth), dim=(1, 2)) + eps

        # 在实现的时候，往往会加上一个smooth，防止分母为0的情况出现
        dices_sub = 2 * (intersection + 1) / (union + 1)
        dices_bs[idx] = dices_sub

    dices = torch.mean(dices_bs, dim=0)
    dice  = torch.mean(dices)
    dice_loss = 1 - dice

    # 最后越小越好
    return dice_loss

class DiceLoss(nn.Module):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, weight=None, num_class=2):
        super(DiceLoss, self).__init__()
 
    def forward(self, pred, label, eps=1e-8):
        return AchieveDice_1(pred, label)
