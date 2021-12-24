'''
Author: zhonzxad
Date: 2021-06-24 10:10:02
LastEditTime: 2021-12-02 21:24:45
LastEditors: zhonzxad
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super(UNetConv2, self).__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):  # 左闭右开区间
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                # setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # # initialise the blocks
        # 统一放置到最外层进行初始化
        # for m in self.children():
        #     init_weight(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(UNetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = UNetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # # initialise the blocks
        # 统一放置到最外层进行初始化
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weight(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print("inputs0 shape is {}".format(inputs0.shape))
        outputs0 = self.up(inputs0)
        # print("outputs0 shape is {}".format(outputs0.shape))
        for i in range(len(input)):
            # print("outputs0 shape is {}, input shape is {}".format(
            #    outputs0.shape, input[i].shape))
            outputs0 = torch.cat([outputs0, input[i]], 1) 
            # print("outputs0 shape is {}".format(outputs0.shape))
        return self.conv(outputs0)

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = UNetUp(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = UNetUp(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # 统一放置到最外层进行初始化
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weight(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
