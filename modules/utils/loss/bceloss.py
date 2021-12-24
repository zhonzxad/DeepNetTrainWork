'''
Author: zhonzxad
Date: 2021-06-24 10:10:02
LastEditTime: 2021-11-25 16:10:24
LastEditors: zhonzxad
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def AchieveBCE_1(pred, label):
    """
    使用于网络输出结果和标签图都是四维的情况
    比如标签图是One-Hot形式的
    """
    n, c, h, w = pred.size()
    nt, ht, wt, ct = label.size()
    # 如果输出结果与原始结果size不同 双线性插值
    if h != ht and w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

    label = label.permute(0, 3, 1, 2)

    bce_loss = nn.BCELoss(size_average=True)(pred, label)
    
    return bce_loss

def AchieveBCE_2(pred, label):
    bce_loss = nn.BCELoss(size_average=True)
    probs = torch.sigmoid(pred)  # 二分类问题，sigmoid等价于softmax
    probs_flat = probs.view(-1)
    targets_flat = label.view(-1)
    return bce_loss(probs_flat, targets_flat)

def clip_by_tensor(t, t_min, t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def AchieveBCE_3(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)  # 不要 0和1,要么是接近0的数，要么是接近1的数
    print("img shape is {}, png shape is {}".format(pred.shape, target.shape))
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

def AchieveBCE_4(pred, label):
    selfpred = torch.squeeze(pred, dim=1)
    print("img shape is {}, png shape is {}".format(selfpred.shape, label.shape))
    bce_loss = nn.BCELoss(size_average=True)(selfpred, label)
    #print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_loss

def AchieveBCE_5(predict, target):
    # chanel  = predict.shape[1]
    # predict  = torch.sigmoid(predict)      # 二分类问题，sigmoid等价于softmax
    # 断开这两个变量之间的依赖,深拷贝
    # predict  = predict.transpose(1, 2).transpose(2, 3).contiguous().view(-1, chanel)
    # targets = torch.stack([targets, targets.clone()], dim=1)
    # targets = targets.view(-1, chanel)
    # predict = predict.squeeze(dim=1)
    target = target.permute([0, 3, 1, 2])

    bceloss = nn.BCELoss()(predict, target)

    return bceloss

def AchieveBCE_6(pred, label):
    """
    使用于网络输出结果是四维，但是标签是三维的情况
    比如标签就是读入的灰度图
    """
    n, c, h, w = pred.size()
    nt, ht, wt = label.size()
    # 如果输出结果与原始结果size不同 双线性插值
    if h != ht and w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

    pred = pred.view(n, h, w)

    bce_loss = nn.BCELoss()(pred, label)
    return bce_loss


def AchieveBCE_7(pred, label):
    """
    使用于网络输出结果是四维，但是标签是三维的情况
    比如标签就是读入的灰度图
    使用BCEWithLogitsLoss不需要经过softmax
    """
    n, c, h, w = pred.size()
    nt, ht, wt, ct = label.size()
    # 如果输出结果与原始结果size不同 双线性插值
    if h != ht and w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)


    pred = pred.contiguous().view(n, c, -1)
    label = label.permute(0, 3, 1, 2)  # 全部调换为 [n, c, h, w]
    label = label.contiguous().view(n, c, -1)

    bce_loss = nn.BCEWithLogitsLoss()(pred, label)

    return bce_loss

# BCELOSS适用于多标签问题，比如有猫有狗之类 ??不确定
def BCE_loss(pred, label):
    '''label
    BCEloss外部计算接口
    pred:网络预测图
    label：标签图
    '''
    bceloss = AchieveBCE_1(pred, label)
    return bceloss

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()

    # BCEWithLogitsLoss函数包括了 Sigmoid 层和 BCELoss 层. 适用于多标签分类任务
    # 如果你的网络的最后一层不是sigmoid，你需要把BCELoss换成BCEWithLogitsLoss
 
    def forward (self, predict, target):
        
        return AchieveBCE_7(predict, target)
