# -*- coding: utf-8 -*-
# @Time     : 2022/2/27 下午 09:36
# @Author   : zhonzxad
# @File     : attention_layer.py
import torch
from torch import nn
import math
import torch.nn.functional as F

class SELayer(nn.Module):
    """SE注意力机制
    Squeeze.将H×W×C*H的特征图压缩为1×1×C ，一般是用global average pooling实现
    Excitation.得到Squeeze的1×1×C的特征图后，使用FC全连接层，对每个通道的重要性进行预测，得到不同channel的重要性大小。有两个全连接，一个降维，一个恢复维度。
    Scale.这里Scale就是一个简单的加权操作,最后将学习到的各个channel的激活值（sigmoid激活，值0~1）乘以之前的feature map的对应channel上。
    """
    def __init__(self, in_chanel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(in_chanel, in_chanel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_chanel // reduction, in_chanel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape # x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Flatten(nn.Module):
    """按照第一维展平"""
    def __init__(self):
        super(Flatten, self).__init__()
        pass

    def forward(self, x):
        batch = x.shape[0]
        ret = x.view(batch, -1)
        return ret

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear

        self.MLP_linner = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )
        self.MLP_Conv2D = nn.Sequential(
            # bias 即是否要添加偏置参数作为可学习参数的一个
            nn.Conv2d(input_channels, input_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_channels // reduction_ratio, input_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        # 在通道注意力机制中使用全连接层来
        CBAM_use_linner = True
        if CBAM_use_linner:
            out = self.MLP_linner(avg_values) + self.MLP_linner(max_values)
            scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            out = self.MLP_Conv2D(avg_values) + self.MLP_Conv2D(max_values)
            scale = self.sigmoid(out)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用大小为3/7的卷积核，为了更大的感受野，一般用7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        # 原版的做法是只使用激活函数作为恢复特征，新版做法增加bn层
        CBAM_use_newfun = True
        if CBAM_use_newfun:
            out = self.bn(out)
            # scale = x * torch.sigmoid(out)
        else:
            # scale = x * self.sigmoid(out)
            pass

        return out

class CBAM(nn.Module):
    """CBAM(convolutional block attention modules)是一个卷积块注意力模块
    做用于输入图像，按照顺序将注意力机制应用于通道，而后是空间维度。
    CBAM的结果是一个加权的特征图，考虑了输入图像的通道和空间区域。
    """

    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        chanel_out = self.channel_att(x)
        space_out = self.spatial_att(chanel_out)

        out = space_out * chanel_out

        return out

class simam(torch.nn.Module):
    """无参注意力机制SimAM
    code from：https://github.com/ZjjConan/SimAM
    info：https://cloud.tencent.com/developer/article/1854055?from=article.detail.1919484
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam, self).__init__()

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

class GAM(nn.Module):
    """
    GAM 全局注意力机制(GAM)，相教于传统通道注意力和空间注意力能更好的涨点。
    CBAM依次进行通道和空间注意力操作，而BAM并行进行。但它们都忽略了通道与空间的相互作用，从而丢失了跨维信息
    GAM 一种注意力机制能够在减少信息弥散的情况下也能放大全局维交互特征
    引自：Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions
    """

    def __init__(self, in_channels, out_channels=None, rate=4):
        super(GAM, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)    # b,c,h,w

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

class h_swish(nn.Module):
    """Coordinate Attention中特殊激活函数
    non-linear
    h_swish激活函数,非线性激活函数
    性能比肩Relu
    """
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x

class CoorAtt_User(nn.Module):
    """Coordinate Attention 特殊实现版
    info：https://blog.csdn.net/practical_sharp/article/details/115789065
    """
    def __init__(self, in_channels, out_channels, reduction = 32):
        super(CoorAtt_User, self).__init__()
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1,None))

        middle = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, middle, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(middle, out_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # inout x shape [b, c, h, w]
        identity = x
        batch_size, c, h, w = x.size()  # [b, c, h, w]

        # X Avg Pool
        x_h = self.poolh(x)    # [b, c, h, 1]
        # Y Avg Pool
        x_w = self.poolw(x)        # [b, c, 1, w]
        x_w = x_w.permute(0, 1, 3, 2) # [b, c, w, 1]

        # following the paper, cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        y = torch.cat((x_h, x_w), dim=2)   # [batch_size, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]

        # split
        x_h, x_w = torch.split(y, [h,w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        x_w = x_w.permute(0, 1, 3, 2) # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))

        # re-weight
        out = identity * attention_h * attention_w
        return out

class CoordAtt(nn.Module):
    """协同注意力官方实现
    code from: https://github.com/Andrew-Qibin/CoordAttention
    info：https://cloud.tencent.com/developer/article/1829677?from=article.detail.1919484
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out