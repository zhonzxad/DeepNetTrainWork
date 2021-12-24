# -*- coding: utf-8 -*-
# @Time     : 2021/12/23 下午 04:19
# @Author   : zhonzxad
# @File     : init_weight.py

import torch
import torch.nn as nn
import math

class init_weight():
    """
    权重初始化相关类
    net.modules()迭代的返回: AlexNet,Sequential,Conv2d,ReLU,MaxPool2d,LRN,AvgPool3d....,Conv2d,...,Conv2d,...,Linear,
    这里,只有Conv2d和Linear才有参数
    net.children()只返回实际存在的子模块: Sequential,Sequential,Sequential,Sequential,Sequential,Sequential,Sequential,Linear
    """

    def init_func(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if self.init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, self.init_gain)
            elif self.init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=self.init_gain)
            elif self.init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=self.init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def init_func_2(self, model):
        #权值参数初始化
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if self.init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, math.sqrt(2. / n))
                elif self.init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2. / n))
                elif self.init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, net, init_type='normal', init_gain=0.02):
        super(init_weight, self).__init__()
        self.net = net
        self.init_type = init_type
        self.init_gain = init_gain

    def init(self):
        self.net.apply(self.init_func_2)

