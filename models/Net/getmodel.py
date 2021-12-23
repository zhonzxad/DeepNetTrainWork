'''
Author: zhonzxad
Date: 2021-11-23 09:49:29
LastEditTime: 2021-12-17 21:38:26
LastEditors: zhonzxad
'''
# import os
# import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
import argparse

import torch
import torch.nn as nn
from loguru import logger
from models.Net.FCN.fcn import FCN
from models.Net.ResNet.ResNet import GetResNet
from models.Net.ResNet.resnet18 import RestNet18
from models.Net.SegNet.SegNet import SegNet
from models.Net.Attention_UNet.Attention_UNet import At_UNet
from models.Net.UNet.UNet import UNet
from models.Net.UNet.UNet_2Plus import UNet_2Plus
from models.Net.UNet.UNet_3Plus import UNet_3Plus
from models.Net.UNet.UNetBili import UNetVGG16

from models.Net.Signal_model.init_weight import init_weight

# from .FCN.fcn import FCN
# from .ResNet.ResNet import GetResNet
# from .ResNet.resnet18 import RestNet18
# from .SegNet.SegNet import SegNet
# from .SmaAtUNer.SmaAt_UNet import SmaAtUNet
# from .UNet.UNet import UNet
# from .UNet.UNet_2Plus import UNet_2Plus
# from .UNet.UNet_3Plus import UNet_3Plus
# from .UNet.UNetBili import UNetVGG16

class GetModel():
    def __init__(self, args):
        if type(args) == argparse.Namespace:
            self.args    = args
            self.IMGSIZE = self.args.IMGSIZE
            self.NCLASS  = self.args.nclass
        else:
            self.IMGSIZE = args[0]
            self.NCLASS  = args[1]


    def Createmodel(self, is_train=True):
        # 加载模型
        # model = UNet(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        # model = UNetVGG16(num_classes=CLASSNUM, in_channels=IMGSIZE[2])
        # model = UNet_2Plus(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model = RestNet18(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model   = SegNet(input_channels=IMGSIZE[2], num_class=CLASSNUM)
        # model   = FCN(input_channels=IMGSIZE[2], num_class=CLASSNUM)
        model = At_UNet(n_channels=self.IMGSIZE[2], n_classes=self.NCLASS)

        # # 根据是否为训练集设置训练
        # if is_train:
        #     model.train()
        # else:
        #     model.eval()

        return model

    def init_weights(self, model, type: str = "kaiming"):
        assert type in [
            "kaiming",
            "normal",
            "xavier",
            "orthogonal",
        ], "Input device is not valid"

        # 初始化网络相关权重
        init_weight(model, type).init()

        # if loger is not None:
        #     loger.write("使用{}方法初始化网络相关权重".format(init_type))
        # else:
        #     print("使用{}方法初始化网络相关权重".format(init_type))
        logger.success("使用{}方法初始化网络相关权重".format(type))