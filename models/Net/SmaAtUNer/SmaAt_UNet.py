'''
Author: zhonzxad
Date: 2021-12-02 16:55:18
LastEditTime: 2021-12-09 20:39:28
LastEditors: zhonzxad
https://www.shangmayuan.com/a/2d07edb726594dd9a18e1832.html
https://github.com/HansBambel/SmaAt-UNet/blob/master/models/SmaAt_UNet.py
'''
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
# sys.path.append(BASE_DIR)
"""Notion
对于标准的k*k的卷积核，stride为s，分三种情况分析：
1）s > 1 在卷积同时并伴随了downsampling操作，卷积后图像变小。
2）s = 1 在padding为SAME时卷积后图像大小不变
3）s < 1 fractionally strided convolution,相当于对原图先作了upsampling操作扩大原图，然后再卷积，这样得到的结果图会变大。
"""

from torch import nn

from SmaAtUNer.SmartLayer import CBAM, GAM
# from SmaAtUNer.unet_parts import OutConv
from SmaAtUNer.unet_parts_DS import DoubleConvDS, DownDS, UpDS, UNetUp_Tradition
from SmaAtUNer.unet_parts_DS import InConv, OutConv


class SmaAtUNet(nn.Module):
    """增加转置卷积等"""

    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAtUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio

        # 1*1的卷积升/降特征的维度, 这里的维度指的是通道数(厚度),而不改变图片的宽和高
        self.Conv2D_1_In = InConv(self.n_channels, 64, k_size=1)
        # self.Conv2D_3_In = InConv(self.n_channels, 64, k_size=3)
        # self.DCS_In      = DoubleConvDS(self.n_channels, 64, kernels_per_layer=self.kernels_per_layer)

        # self.cbam1 = CBAM(64, reduction_ratio=self.reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=self.kernels_per_layer)
        # self.cbam2 = CBAM(128, reduction_ratio=self.reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=self.kernels_per_layer)
        # self.cbam3 = CBAM(256, reduction_ratio=self.reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=self.kernels_per_layer)
        # self.cbam4 = CBAM(512, reduction_ratio=self.reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=self.kernels_per_layer)
        # self.cbam5 = CBAM(1024 // factor, reduction_ratio=self.reduction_ratio)

        self.gam1  = GAM(64)
        self.gam2  = GAM(128)
        self.gam3  = GAM(256)
        self.gam4  = GAM(512)
        self.gam5  = GAM(1024 // factor)

        self.up1_DS = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up2_DS = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up3_DS = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up4_DS = UpDS(128, 64, self.bilinear, kernels_per_layer=self.kernels_per_layer)

        # self.up1_Conv = UNetUp_Tradition(1024, 512)
        # self.up2_Conv = UNetUp_Tradition(512, 256)
        # self.up3_Conv = UNetUp_Tradition(256, 128)
        # self.up4_Conv = UNetUp_Tradition(128, 64)

        self.out_DS = OutConv(64, self.n_classes)
        # self.out_Conv = OutConv(64 // 2, self.n_classes)

    def forward(self, inputs):
        x = inputs
        x1 = self.Conv2D_1_In(x)

        x1_att = self.gam1(x1)
        x2 = self.down1(x1)
        x2_att = self.gam2(x2)
        x3 = self.down2(x2)
        x3_att = self.gam3(x3)
        x4 = self.down3(x3)
        x4_att = self.gam4(x4)
        x5 = self.down4(x4)
        x5_att = self.gam5(x5)

        x = self.up1_DS(x5_att, x4_att)
        x = self.up2_DS(x, x3_att)
        x = self.up3_DS(x, x2_att)
        x = self.up4_DS(x, x1_att)

        logits = self.out_DS(x)

        return logits