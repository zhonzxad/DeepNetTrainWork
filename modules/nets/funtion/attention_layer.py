# -*- coding: utf-8 -*-
# @Time     : 2022/2/27 下午 09:36
# @Author   : zhonzxad
# @File     : attention_layer.py
import torch
from torch import nn
import math
import torch.nn.functional as F

class h_swish(nn.Module):
    """h_swish激活函数
    性能比肩Relu
    """
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x

class simam_module(torch.nn.Module):
    """无参注意力机制SimAM
    code from：https://github.com/ZjjConan/SimAM
    info：https://cloud.tencent.com/developer/article/1854055?from=article.detail.1919484
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    def forward(self, x):
        e_lambda = 1e-4
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / \
            (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + e_lambda)) + \
            0.5

        return x * self.activaton(y)

class CA_Block(nn.Module):
    """Coordinate Attention
    code from: https://github.com/Andrew-Qibin/CoordAttention
    info：https://cloud.tencent.com/developer/article/1829677?from=article.detail.1919484
    """
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction,
                                  kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel,
                             kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel,
                             kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class CoorAttention(nn.Module):
    """Coordinate Attention 特殊实现版
    info：https://blog.csdn.net/practical_sharp/article/details/115789065
    """
    def __init__(self,in_channels, out_channels, reduction = 32):
        super(CoorAttention, self).__init__()
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1,None))
        middle = max(8, in_channels//reduction)
        self.conv1 = nn.Conv2d(in_channels,middle,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x): # [batch_size, c, h, w]
        identity = x
        batch_size, c, h, w = x.size()  # [batch_size, c, h, w]
        # X Avg Pool
        x_h = self.poolh(x)    # [batch_size, c, h, 1]

        #Y Avg Pool
        x_w = self.poolw(x)    # [batch_size, c, 1, w]
        x_w = x_w.permute(0,1,3,2) # [batch_size, c, w, 1]

        #following the paper, cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        y = torch.cat((x_h, x_w), dim=2)   # [batch_size, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
        # split
        x_h, x_w = torch.split(y, [h,w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        x_w = x_w.permute(0,1,3,2) # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))
        # re-weight
        return identity * attention_h * attention_w

