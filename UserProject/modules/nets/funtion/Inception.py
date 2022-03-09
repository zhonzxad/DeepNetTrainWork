# -*- coding: utf-8 -*-
# @Time     : 2021/12/18 上午 10:06
# @Author   : zhonzxad
# @File     : Inception.py
import torch
import torch.nn.functional as F
from torch import nn

"""Notion
https://www.cnblogs.com/leebxo/p/10315490.html
"""

class BasicConv2d(nn.Module):

    def __init__(self, in_cha, out_cha, kernel_size, stride=1, padding=1):
        super(BasicConv2d, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, kernel_size=kernel_size),
            nn.BatchNorm2d(out_cha),
        )

    def forward(self, _in):
        ret = self.Conv(_in)
        return ret

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1) # 1

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Inception(nn.Module):
    """CSDN Inception Test"""

    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool):
        # inputSize:输入尺寸
        # kernelSize:第1步骤和第2步骤中第二个卷积核的尺寸，是一个列表
        # kernelStride:同上
        # outputSize:同上
        # reduceSize:1*1卷积中的输出尺寸，是一个列表
        # pool: 是一个池化层
        super(Inception, self).__init__()
        self.layers = {}
        poolFlag = True
        fname = 0
        for p in kernelSize, kernelStride, outputSize, reduceSize:
            if len(p) == 4:
                (_kernel, _stride, _output, _reduce) = p
                self.layers[str(fname)] = nn.Sequential(
                    # Convolution 1*1
                    nn.Conv2d(inputSize, _reduce, 1),
                    nn.BatchNorm2d(_reduce),
                    nn.ReLU(),
                    # Convolution kernel*kernel
                    nn.Conv2d(_reduce, _output, _kernel, _stride),
                    nn.BatchNorm2d(_output),
                    nn.ReLU())
            else:
                if poolFlag:
                    assert len(p) == 1
                    self.layers[str(fname)] = nn.Sequential(
                        # pool
                        pool, #这里的输出尺寸需要考虑一下
                        nn.Conv2d(inputSize, p, 1),
                        nn.BatchNorm2d(p),
                        nn.ReLU())
                    poolFlag = False
                else:
                    assert len(p) == 1
                    self.layers[str(fname)] = nn.Sequential(
                        # Convolution 1*1
                        nn.Conv2d(inputSize, p, 1),
                        nn.BatchNorm2d(p),
                        nn.ReLU())
            fname += 1

        if poolFlag:
            self.layers[str(fname)] = nn.Sequential(pool)
            poolFlag = False

    def forward(self, x):
        for key, layer in self.layers.items:
            if key == str(0):
                out = layer(x)
            else:
                out = torch.cat((out, layer(x)), 1) #因为有Batch，所以是在第1维方向串接。
        return out