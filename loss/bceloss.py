import torch
import torch.nn as nn
import torch.nn.functional as F


def AchieveBCE_1(pred, label):
    n, c, h, w = pred.size()
    nt, ht, wt = label.size()
    # 如果输出结果与原始结果size不同 双线性插值
    if h != ht and w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

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

# BCELOSS适用于多标签问题，比如有猫有狗之类 ??不确定
def BCE_loss(pred, label):
    '''
    BCEloss外部计算接口
    pred:网络预测图
    label：标签图
    '''
    bceloss = AchieveBCE_1(pred, label)
    return bceloss

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
 
    def forward (self, logits, targets):
        chanel  = logits.shape[1]
        logits  = torch.sigmoid(logits)      # 二分类问题，sigmoid等价于softmax
        logits  = logits.transpose(1, 2).transpose(2, 3).contiguous().view(-1, chanel)
        targets = torch.stack([targets, targets.clone()], dim=1)
        targets = targets.view(-1, chanel)
        bceloss = nn.BCELoss(size_average=True)(logits, targets)
        return bceloss
