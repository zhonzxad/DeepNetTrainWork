'''
Author: zhonzxad
Date: 2021-12-02 16:55:18
LastEditTime: 2021-12-17 21:37:07
LastEditors: zhonzxad
https://www.shangmayuan.com/a/2d07edb726594dd9a18e1832.html
https://github.com/HansBambel/SmaAt-UNet/blob/master/models/SmaAt_UNet.py
'''
# import os
# import sys
# BASE_DIR = os._path.dirname(os._path.dirname(os._path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
# sys._path.append(BASE_DIR)
"""Notion
对于标准的k*k的卷积核，stride为s，分三种情况分析：
1）s > 1 在卷积同时并伴随了downsampling操作，卷积后图像变小。
2）s = 1 在padding为SAME时卷积后图像大小不变
3）s < 1 fractionally strided convolution,相当于对原图先作了upsampling操作扩大原图，然后再卷积，这样得到的结果图会变大。
"""
from torch import nn

from UserProject.modules.nets.funtion.attention_layer import CoorAtt_User
from UserProject.modules.nets.DefectUNet.unet_parts_conv import DoubleDNR, DownDS
from UserProject.modules.nets.DefectUNet.unet_parts_conv import OutConv, UpDS


class DefectUNet(nn.Module):
    """DefectUNet语义分割检测网络
    """

    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(DefectUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio

        # 1*1的卷积升/降特征的维度, 这里的维度指的是通道数(厚度),而不改变图片的宽和高
        # self.Conv2D_1_In = InConv(self.n_channels, 64, k_size=1)
        # self.Conv2D_3_In = InConv(self.n_channels, 64, k_size=3)
        self.DCS_In      = DoubleDNR(self.n_channels, 64, kernels_per_layer=self.kernels_per_layer)

        factor = 2 if self.bilinear else 1

        self.att_mode_1 = CoorAtt_User(64)
        self.att_mode_2 = CoorAtt_User(128)
        self.att_mode_3 = CoorAtt_User(256)
        self.att_mode_4 = CoorAtt_User(512)
        self.att_mode_5 = CoorAtt_User(1024 // factor)

        # self.att_mode_1 = CBAM(64, reduction_ratio=self.reduction_ratio)
        # self.att_mode_2 = CBAM(128, reduction_ratio=self.reduction_ratio)
        # self.att_mode_3 = CBAM(256, reduction_ratio=self.reduction_ratio)
        # self.att_mode_4 = CBAM(512, reduction_ratio=self.reduction_ratio)
        # self.att_mode_5 = CBAM(1024 // factor, reduction_ratio=self.reduction_ratio)

        # self.down1 = DownDS(64, 128, kernels_per_layer=self.kernels_per_layer)
        # self.down2 = DownDS(128, 256, kernels_per_layer=self.kernels_per_layer)
        # self.down3 = DownDS(256, 512, kernels_per_layer=self.kernels_per_layer)
        # self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=self.kernels_per_layer)

        self.down1 = DownDS(64, 128, kernels_per_layer=self.kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=self.kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=self.kernels_per_layer)
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=self.kernels_per_layer)

        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=self.kernels_per_layer)

        # self.up1 = UNetUp_Tradition(1024, 512)
        # self.up2 = UNetUp_Tradition(512, 256)
        # self.up3 = UNetUp_Tradition(256, 128)
        # self.up4 = UNetUp_Tradition(128, 64)

        self.out = OutConv(64, self.n_classes)
        # self.out = OutConv(64 // 2, self.n_classes)

    def forward(self, x):
        x1 = self.DCS_In(x)

        # x1_att = self.att_mode_1(x1)
        # x2 = self.down1(x1)
        # x2_att = self.att_mode_2(x2)
        # x3 = self.down2(x2)
        # x3_att = self.att_mode_3(x3)
        # x4 = self.down3(x3)
        # x4_att = self.att_mode_4(x4)
        # x5 = self.down4(x4)
        # x5_att = self.att_mode_5(x5)

        x1_att = self.att_mode_1(x1)
        x2 = self.down1(x1_att)
        x2_att = self.att_mode_2(x2)
        x3 = self.down2(x2_att)
        x3_att = self.att_mode_3(x3)
        x4 = self.down3(x3_att)
        x4_att = self.att_mode_4(x4)
        x5 = self.down4(x4_att)
        x5_att = self.att_mode_5(x5)

        out = self.up1(x5_att, x4_att)
        out = self.up2(out, x3_att)
        out = self.up3(out, x2_att)
        out = self.up4(out, x1_att)

        ret = self.out(out)

        return ret
