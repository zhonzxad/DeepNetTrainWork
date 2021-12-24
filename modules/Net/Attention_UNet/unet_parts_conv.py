'''
Author: zhonzxad
Date: 2021-12-02 16:55:18
LastEditTime: 2021-12-17 21:39:28
LastEditors: zhonzxad
'''
""" Parts of the U-Net model """
# Base model taken from: https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.Net.Net_funtion.layer import GroupNorm, GropConv, DilConv
from modules.Net.Attention_UNet.Attention_Layer import DepthwiseSeparableConv


class DoubleConvDS(nn.Module):
    """(convolution(深度可分离卷积) => [批处理归一化] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            # DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            DilConv(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(mid_channels),
            GroupNorm(mid_channels),
            nn.ReLU(inplace=True),

            # DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            DilConv(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=2),
            # nn.BatchNorm2d(out_channels),
            GroupNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv
    使用maxpool将特征图缩小之后采用通道可分离卷积
    maxpool提取重要信息的操作，可以去掉不重要的信息，减少计算开销(改变图像的大小)
    maxpool：引自，https://blog.csdn.net/L1778586311/article/details/112159479
    """

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpDS(nn.Module):
    """Upscaling then double conv
    上采样之后进行一个 双倍 深度可分离卷积
    """

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        # 如果是双线性的，使用普通卷积来减少通道数
        if bilinear:
            # 根据scale_factor指定的上采样倍数及mode采样方式
            # 或者直接使用nn.UpsamplingBilinear2d
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is b*C*H*W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # F.pad,tensor扩充函数,便于对数据集图像或中间层特征进行维度扩充
        # 第二个参数pad依据输入的值对x1进行拓展，默认使用 常量0来进行填充
        # (左边填充数， 右边填充数， 上边填充数， 下边填充数， 前边填充数，后边填充数)，本例填充二维
        x1 = F.pad(x1, pad=[diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode="constant", value=0)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # 这是因为原论文鼓励无填充的下采样与无零填充的上采样一样，可以避免对语义信息的破坏。这也是提出重叠瓦片策略的原因之一
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class InConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_size):
        super(InConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size)

    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetConv2_Tradition(nn.Module):
    """上采样后的卷积池化部分"""

    def __init__(self, in_size, out_size, is_batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super(UNetConv2_Tradition, self).__init__()
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

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class UNetUp_Tradition(nn.Module):
    """使用传统方式进行卷积上采样"""

    def __init__(self, in_size, out_size, is_deconv=True, n_concat=2):
        super(UNetUp_Tradition, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = UNetConv2_Tradition(in_size, out_size // 2, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

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
