'''
Author: zhonzxad
Date: 2021-11-16 15:09:11
LastEditTime: 2021-11-16 21:12:52
LastEditors: zhonzxad
'''
import torch
import torch.nn as nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        extra_x = self.extra(x)
        output  = self.conv1(x)
        out     = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super(RestNet18, self).__init__()
        self.nclass = n_classes
        self.inchannel = in_channels

        self.conv1 = nn.Conv2d(self.inchannel, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, self.nclass)

        self.changeout = nn.Conv2d(512, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(x.shape[0], -1)
        # out = self.fc(out)
        return out
