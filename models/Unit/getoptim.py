'''
Author: zhonzxad
Date: 2021-11-22 20:22:53
LastEditTime: 2021-11-29 20:18:29
LastEditors: zhonzxad
'''
import torch

'''自动优化器参数：
    mode(str)模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， 
    max 表示当指标不再升高(如监测 accuracy)。
    factor(float)学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
    patience(int)忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
    verbose(bool) 向控制台输出一条学习率调整的信息， 默认false
    threshold_mode(str)
        选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
        当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
        当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
        当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
        当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
        threshold(float)
        配合 threshold_mode 使用。
    cooldown(int) “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
    min_lr(float or list)学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
    eps(float)学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
    '''

# 设置学习率
def UserOptim1(model, lr):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-07)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", \
                                                            verbose = True, \
                                                            factor=0.5, patience=1, cooldown=1, \
                                                            eps=1e-7)

    return optimizer, scheduler

def UserOptim2(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    return optimizer, scheduler

def UserOptim3(model, lr):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    return optimizer, scheduler

def UserOptim4(model, lr, momentum=0.9, weight_decay=0.001, lr_gamma=0.001, lr_decay=0.75):
    """
    从迁移学习框架库中学习到的优化器用法
    lr: 0.01
    weight_decay: 0.001 # 5e-4
    momentum: 0.9
    lr_scheduler: True
    lr_gamma: 0.001
    lr_decay: 0.75
    """
    initial_lr = lr if lr is not None else 1.0
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, \
                            lambda x: lr * (1. + lr_gamma * float(x)) ** (-lr_decay))

    return optimizer, scheduler

def UserOptim5(model, lr, lr_gamma=0.5, lr_decay=0.75):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, \
                            lambda x: lr * (1. + lr_gamma * float(x)) ** (-lr_decay))

    return optimizer, scheduler

def CreateOptim(model, lr):

    return UserOptim3(model, lr)
