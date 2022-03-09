# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet_layers import UNetConv2, UNetUp

from UserProject.modules.nets.DefectUNet.unet_parts_conv import DoubleDNR, DownDS
from UserProject.modules.nets.DefectUNet.unet_parts_conv import OutConv, UpDS
from UserProject.modules.nets.funtion.attention_layer import CoorAtt_User
from UserProject.modules.nets.funtion.layer import GroupNorm


class UNet(nn.Module):

    def __init__(self, input_channels=3, num_class=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = input_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nclass = num_class
        #
        #filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # 下采样
        self.conv1 = UNetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.maxpool1 = nn.Sequential(GroupNorm(self.in_channels))
        # self.maxpool1 = nn.Sequential(CoorAtt_User(64),
        #                               GroupNorm(self.in_channels),
        #                               # nn.MaxPool2d(kernel_size=2),
        #                               )

        self.conv2 = UNetConv2(filters[0], filters[1], self.is_batchnorm)
        # self.conv2 = DownDS(filters[0], filters[1], 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.maxpool2 = nn.Sequential(GroupNorm(filters[0]))
        # self.maxpool2 = nn.Sequential(CoorAtt_User(128),
        #                               GroupNorm(filters[0]),
        #                               # nn.MaxPool2d(kernel_size=2)
        #                               )

        self.conv3 = UNetConv2(filters[1], filters[2], self.is_batchnorm)
        # self.conv3 = DownDS(filters[1], filters[2], 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.maxpool3 = nn.Sequential(GroupNorm(filters[1]))
        # self.maxpool3 = nn.Sequential(CoorAtt_User(256),
        #                               GroupNorm(filters[1]),
        #                               # nn.MaxPool2d(kernel_size=2)
        #                               )

        self.conv4 = UNetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.conv4 = DownDS(filters[2], filters[3], 2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # self.maxpool4 = nn.Sequential(GroupNorm(filters[2]))
        # self.maxpool4 = nn.Sequential(CoorAtt_User(512),
        #                               GroupNorm(filters[2]),
        #                               # nn.MaxPool2d(kernel_size=2)
        #                               )

        self.center = UNetConv2(filters[3], filters[4], self.is_batchnorm)
        # self.center = DownDS(filters[3], filters[4], 2)

        # 
        self.up_up   = nn.ConvTranspose2d(filters[4], filters[4] // 2, kernel_size=4, stride=2, padding=1)

        # 上采样恢复
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv)
        # self.up_concat4 = UpDS(filters[4], filters[3], 2)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat4 = UpDS(filters[3], filters[2], 2)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv)
        # self.up_concat4 = UpDS(filters[2], filters[1], 2)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv)
        # self.up_concat4 = UpDS(filters[1], filters[0], 2)

        #
        self.outconv1 = nn.Conv2d(filters[0], self.nclass, kernel_size=1)

        # # 初始化权重,放到getmodel中统一进行初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weight(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weight(m, init_type='kaiming')

    # 内积
    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)

        return final

    def forward(self, inputs):
        # print("\n inputs shape is {}".format(inputs.shape))
        conv1 = self.conv1(inputs)
        # print("\n conv1 shape is {}".format(conv1.shape))
        maxpool1 = self.maxpool1(conv1)
        # print("\n maxpool1 shape is {}".format(maxpool1.shape))

        conv2 = self.conv2(maxpool1)
        # print("\n conv2 shape is {}".format(conv2.shape))
        maxpool2 = self.maxpool2(conv2)
        # print("\n maxpool2 shape is {}".format(maxpool2.shape))

        conv3 = self.conv3(maxpool2)
        # print("\n conv3 shape is {}".format(conv3.shape))
        maxpool3 = self.maxpool3(conv3)
        # print("\n maxpool3 shape is {}".format(maxpool3.shape))

        conv4 = self.conv4(maxpool3)
        # print("\n conv4 shape is {}".format(conv4.shape))
        maxpool4 = self.maxpool4(conv4)
        # print("\n maxpool4 shape is {}".format(maxpool4.shape))

        center = self.center(maxpool4)
        # centernew = center.unsqueeze(1)
        # print("\n centernew shape is {}".format(centernew.shape))
        # print("\n center shape is {}, conv4 shape is {}".format(center.shape, conv4.shape))

        up4 = self.up_concat4(center, conv4)
        # print("\n up4 shape is {}".format(up4.shape))
        up3 = self.up_concat3(up4, conv3)
        # print("\n up3 shape is {}".format(up3.shape))
        up2 = self.up_concat2(up3, conv2)
        # print("\n up2 shape is {}".format(up2.shape))
        up1 = self.up_concat1(up2, conv1)
        # print("\n up1 shape is {}".format(up1.shape))

        d1 = self.outconv1(up1)            
        # print("\n outconv1 shape is {}".format(d1.shape))

        # 激活函数常用于全连接层之后，增加全连接层之后的非线性特征，全连接就是将两个层之间权值之类的全部联系在一起
        # return torch.sigmoid(d1) 
        return d1
