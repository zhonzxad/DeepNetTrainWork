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

from torch import nn

from SmaAtUNer.SmartLayer import CBAM
from SmaAtUNer.unet_parts import OutConv
from SmaAtUNer.unet_parts_depthwise_separable import DoubleConvDS, DownDS, UpDS


class SmaAtUNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAtUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio

        # 1*1的卷积升/降特征的维度, 这里的维度指的是通道数(厚度),而不改变图片的宽和高
        self.Conv2D_1_In = nn.Conv2d(self.n_channels, 64, kernel_size=1)
        self.Conv2D_3_In = nn.Conv2d(self.n_channels, 64, kernel_size=3)
        self.DCS_In      = DoubleConvDS(self.n_channels, 64, kernels_per_layer=self.kernels_per_layer)

        self.cbam1 = CBAM(64, reduction_ratio=self.reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=self.kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=self.reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=self.kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=self.reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=self.kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=self.reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=self.kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=self.reduction_ratio)

        self.up1_DS = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up2_DS = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up3_DS = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=self.kernels_per_layer)
        self.up4_DS = UpDS(128, 64, self.bilinear, kernels_per_layer=self.kernels_per_layer)

        # self.up1_Conv = UpDS(1024, 512 // factor)
        # self.up2_Conv = UpDS(512, 256 // factor)
        # self.up3_Conv = UpDS(256, 128 // factor)
        # self.up4_Conv = UpDS(128, 64)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.DCS_In(x)

        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)

        x = self.up1_DS(x5Att, x4Att)
        x = self.up2_DS(x, x3Att)
        x = self.up3_DS(x, x2Att)
        x = self.up4_DS(x, x1Att)

        logits = self.outc(x)

        return logits
