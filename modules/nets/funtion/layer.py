# -*- coding: utf-8 -*-
# @Time     : 2021/12/17 下午 09:25
# @Author   : zhonzxad
# @File     : layer.py
import torch
import math
import torch.nn.functional as F
from torch import nn

from modules.nets.CrackUNet.Attention_Layer import DepthwiseSeparableConv

"""Notion
深度可分离卷积和分组卷积的理解
https://blog.csdn.net/weixin_30793735/article/details/88915612 
"""

class GroupNorm(nn.Module):
    """Group Normbalization（GN）
    GN是一种新的深度学习归一化方式，可以替代BN，
    GN优化了BN在比较小的mini-batch情况下表现不太好的劣势
    BN沿着batch维度进行归一化，其受限于BatchSize大小，当其很小时（值小于32），BN会得到不准确的统计估计，会导致模型误差明显增加
        将 Channels 划分为多个 groups，再计算每个 group 内的均值和方法，以进行归一化。
        GB的计算与Batch Size无关，因此对于高精度图片小BatchSize的情况也是非常稳定的
    这是自己代码实现方式与官方的BG实现方式
    更改自：https://github.com/kuangliu/pytorch-groupnorm/blob/master/groupnorm.py
    https://blog.csdn.net/qq_34107425/article/details/107903800
    """

    def __init__(self, num_features_chanel, num_groups_batch=32, usepytorchsimple=False, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features_chanel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features_chanel, 1, 1))
        self.num_groups_batch = num_groups_batch
        self.eps = eps
        self.UsePytorchSimple = usepytorchsimple
        self.simplePytorchGN = torch.nn.GroupNorm(num_channels=num_features_chanel, num_groups=num_groups_batch)

    def forward(self, x):
        b, c, h, w = x.size()

        if self.UsePytorchSimple:
            ret = self.simplePytorchGN(x)
            return ret
        else:
            g = self.num_groups_batch
            assert c % g == 0

            x = x.view(b, g, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x-mean) / (var+self.eps).sqrt()
            x = x.view(b, c, h, w)

            return x * self.weight + self.bias

class GropConv(nn.Module):
    """
    组卷积，Grouped Convolutions
    引自：https://www.cnblogs.com/shine-lee/p/10243114.html
    参考：https://blog.yani.ai/filter-group-tutorial/
    """
    def __init__(self, in_ch, out_ch, groups=8):
        super(GropConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, groups=groups, bias=False)


    def forward(self, _input):
        out = self.conv(_input)
        return out

class DilConv(nn.Module):
    """
    空洞卷积,https://zhuanlan.zhihu.com/p/50369448
    https://blog.csdn.net/u014767662/article/details/88574643
    """
    def __init__(self, in_chane, out_chane, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_chane, in_chane, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=in_chane, bias=False),
            nn.Conv2d(in_chane, out_chane, kernel_size=1, padding=padding, bias=False),
            # nn.BatchNorm2d(out_chane),  # 仍需要进行批处理，但是放到外侧的模块中顺序处理
        )

    def forward(self, x):
        return self.op(x)

def Ghost_Depthwise_Conv(inp, oup, kernel_size=3, stride=1, relu=False):
    Depthwise_Conv = nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )
    return Depthwise_Conv

class GhostModule(nn.Module):
    """Ghost module
    提出一个仅通过少量计算（论文称为cheap operations）就能生成大量特征图的结构Ghost Module
    其主要目的是缩减计算量
    引自：https://arxiv.org/pdf/1911.11907.pdf
    """
    def __init__(self, in_chanel, ou_chanel, kernel_size=1, stride=1, ratio=2, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        self.ou_chanel = ou_chanel
        init_channels = math.ceil(ou_chanel / ratio)  # math.ceil大于或等于 x 的的最小整数
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
                                nn.Conv2d(in_chanel, init_channels, kernel_size=kernel_size,
                                            stride=stride, padding=kernel_size // 2, bias=False),
                                nn.BatchNorm2d(init_channels),
                                nn.ReLU(inplace=True) if relu else nn.Sequential(),)
        # cheap卷积操作，注意利用了分组卷积进行通道分离(也就是论文说的线性变换)
        self.cheap_operation = nn.Sequential(
                                nn.Conv2d(init_channels, new_channels, kernel_size=dw_size,
                                            stride=1, padding=dw_size//2, groups=init_channels, bias=False),
                                nn.BatchNorm2d(new_channels),
                                nn.ReLU(inplace=True) if relu else nn.Sequential(),)

    def forward(self, x):
        x1 = self.primary_conv(x)          # 主要的卷积操作
        x2 = self.cheap_operation(x1)      # cheap变换操作
        out = torch.cat([x1,x2], dim=1)    # 二者cat到一起

        chanel1 = out[:,:self.ou_chanel, :, :]

        return chanel1

class GhostBottleneck(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size=3, stride=2):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]
        hidden_dim = in_chanel // 2

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_chanel, hidden_dim, kernel_size=1, relu=True),
            # dw
            Ghost_Depthwise_Conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, out_chanel, kernel_size=1, relu=False),
        )

        if stride == 1 and in_chanel == out_chanel:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                Ghost_Depthwise_Conv(in_chanel, in_chanel, 3, stride, relu=True),
                nn.Conv2d(in_chanel, out_chanel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chanel),
            )

    def forward(self, x):
        ret_1 = self.conv(x)
        ret_2 = self.shortcut(x)
        out = ret_1 + ret_2
        return out
