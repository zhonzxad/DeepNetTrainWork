'''
Author: zhonzxad
Date: 2021-11-22 14:52:43
LastEditTime: 2021-11-22 16:45:34
LastEditors: zhonzxad
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class FCN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(FCN, self).__init__()
        pretrained_net = resnet34(pretrained=True)
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) # 第一段
        self.stage2 = list(pretrained_net.children())[-4] # 第二段
        self.stage3 = list(pretrained_net.children())[-3] # 第三段

        self.innetwork = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)
        
        # 通道统一
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        # 8倍上采样
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
        
        # 2倍上采样
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

    def forward(self, x):
        #x = self.innetwork(x)
        
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3) # 1/16
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2) # 1/8
        s = s1 + s2

        s = self.upsample_8x(s) # 1/1
        return s

