'''
Author: zhonzxad
Date: 2021-06-24 12:51:18
LastEditTime: 2021-12-01 09:57:28
LastEditors: zhonzxad
'''
'''
Author: zhonzxad
Date: 2021-06-24 12:51:18
LastEditTime: 2021-11-29 19:50:59
LastEditors: zhonzxad
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def AchieveDice_0(input, target, multiclass = True):

    def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]


    def multiclass_dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        for channel in range(input.shape[1]):
            dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

        return dice / input.shape[1]


    def dice_loss(input, target, multiclass=False):
        b, c, h, w = input.size()
        bt, ht, wt, ct = target.size()

        input = F.softmax(input, dim=1).float()
        target = target.permute(0, 3, 1, 2).float()

        fn = multiclass_dice_coeff if multiclass else dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

    return dice_loss(input, target, multiclass)


def AchieveDice_1(input, target, beta=1, smooth=1, num_classes=2):
    b, c, h, w = input.size()
    bt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
    # input = torch.sigmoid(input)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(b, -1, c)
    temp_inputs = torch.softmax(input, -1)   # # [n, h*w, c]
    temp_target = target.view(bt, -1, ct)    # [n, h*w, c]

    #--------------------------------------------#
    #   ??????dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs                       , axis=[0, 1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0, 1]) - tp

    top = ((1 + beta ** 2) * tp + smooth)
    bottom = ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = top / bottom

    dice_loss = 1 - torch.mean(score)

    # ??????????????????
    return dice_loss

def AchieveDice_2(logits, targets, smooth=1):
    '''
    https://github.com/pytorch/pytorch/issues/1249
    smooth ???????????????0?????????
    '''
    b, c, h, w = logits.size()
    bt, ht, wt, ct = targets.size()
    
    probs = torch.softmax(logits, dim=1)

    m1 = probs.view(c, -1)
    m2 = targets.view(c, -1)
    intersection = torch.mul(m1, m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / c
    
    return score

def AchieveDice_3(pred, label, eps=1e-8):
    '''
    :param y_preds: [bs,num_classes,768,1024]
    :param y_truths: [bs,num_calsses,768,1024]
    :param eps:
    :return:
    '''
    bs, c, h, w = pred.size()
    pred = torch.softmax(pred, dim=1)
    pred = pred.permute(0, 2, 3, 1)
    dices_bs = torch.zeros(bs, c)

    for idx in range(bs):
        y_pred  = pred[idx]    # [768,1024,num_classes]
        y_truth = label[idx]   # [768,1024,num_classes]
        intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(0 , 1)) + eps / 2
        union = torch.sum(torch.mul(y_pred, y_pred), dim=(0, 1)) + \
                    torch.sum(torch.mul(y_truth, y_truth), dim=(0, 1)) + eps

        # ??????????????????????????????????????????smooth??????????????????0???????????????
        dices_sub = 2 * (intersection + 1) / (union + 1)
        dices_bs[idx] = dices_sub

    dices = torch.mean(dices_bs, dim=0)
    dice  = torch.mean(dices)
    dice_loss = 1 - dice

    # ??????????????????
    return dice_loss

def AchieveDice_4(input, target, beta=1, smooth=1e-5, eps=1e-8):
    """
    # [h,w,num_classes]
    # [h,w,num_classes]
    """
    b, c, h, w = input.size()
    bt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.perm(0, 3, 1, 2)   #[bs, h, w, c]

    dices_bs = torch.zeros(b, c)

    for idx in range(b):
        # torch.mul????????????????????????????????????
        y_pred  = input[idx]    # [h, w, c]
        y_truth = target[idx]   # [h, w, c]
        intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(0, 1)) + eps / 2
        union = torch.sum(torch.mul(y_pred, y_pred), dim=(0, 1)) + \
                    torch.sum(torch.mul(y_truth, y_truth), dim=(0, 1)) + eps / 2

        # ??????????????????????????????????????????smooth??????????????????0???????????????
        dices_sub = 2 * (intersection + 1) / (union + 1)
        dices_bs[idx] = dices_sub
        
    # input = torch.sigmoid(input)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(b, -1, c)
    temp_inputs = torch.softmax(input, -1)
    temp_target = target.view(bt, -1, ct)

    #--------------------------------------------#
    #   ??????dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs                       , axis=[0, 1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0, 1]) - tp

    top = ((1 + beta ** 2) * tp + smooth)
    bottom = ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = top / bottom

    dice_loss = 1 - torch.mean(score)

    # ??????????????????
    return dice_loss

def AchieveDice_5(input, target, n_class=2, smooth = 1., class_weight=None):

    b, c, h, w = input.size()
    bt, ht, wt, ct = target.size()

    input = torch.softmax(input, dim=1)

    input = input.permute(0, 2, 3, 1)  # ?????????b, h, w, c

    loss = 0.
    for c in range(n_class):
        iflat = input[:, c ].contiguous().view(-1)  # ???????????????class?????????????????????
        tflat = target[:, c].contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        if class_weight == None:
            w = 0.5                   # ????????????class????????????0.5
        else:
            w = class_weight[c]
        
        loss += w * (1 - ((2. * intersection + smooth) /
                            (iflat.sum() + tflat.sum() + smooth)))
    return loss


def Dice_Loss(pred, label):
    '''
    dice???????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ???????????????????????????????????????????????????????????????????????? dice loss ????????????????????????????????????????????????????????????????????????
    pred:???????????????
    label:?????????
    '''
    dice_loss = AchieveDice_3(pred, label)
    return dice_loss

class DiceLoss(nn.Module):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, weight=None, num_class=2):
        super(DiceLoss, self).__init__()
        self.cls_weights = weight
        pass

    # ????????????????????????sigmoid?????????softmax
 
    def forward(self, pred, label):

        out = AchieveDice_1(pred, label)
        return out
