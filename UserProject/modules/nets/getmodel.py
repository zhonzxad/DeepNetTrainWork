'''
Author: zhonzxad
Date: 2021-11-23 09:49:29
LastEditTime: 2021-12-17 21:38:26
LastEditors: zhonzxad
'''
# import os
# import sys
# BASE_DIR = os._path.dirname(os._path.dirname(os._path.abspath(__file__)))
# sys._path.append(BASE_DIR)
import argparse
from loguru import logger

from UserProject.modules.nets.DefectUNet.DefectUNet import DefectUNet

from UserProject.modules.nets.funtion.init_weight import initweight

from UserProject.modules.nets.FCN.fcn import FCN
from UserProject.modules.nets.ResNet.ResNet import GetResNet
from UserProject.modules.nets.ResNet.resnet18 import RestNet18
from UserProject.modules.nets.SegNet.SegNet import SegNet
from UserProject.modules.nets.UNet.UNet import UNet
from UserProject.modules.nets.UNet.UNet_2Plus import UNet_2Plus
from UserProject.modules.nets.UNet.UNet_3Plus import UNet_3Plus
from UserProject.modules.nets.UNet.UNetBili import UNetVGG16
from UserProject.modules.nets.ResUNet.resunet import ResUNet50

class GetModel:

    def __init__(self, args):
        if type(args) == argparse.Namespace:
            self.IMGSIZE = args.IMGSIZE
            self.NCLASS  = args.n_class
        else:
            self.IMGSIZE = args[0]
            self.NCLASS  = args[1]


    def Createmodel(self, is_train=True):
        # 加载模型
        # model = UNet(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        # model = UNetVGG16(num_classes=self.NCLASS, in_channels=self.IMGSIZE[2])
        # model = UNet_2Plus(in_channels=self.IMGSIZE[2], n_classes=self.NCLASS)
        # model = RestNet18(in_channels=self.IMGSIZE[2], n_classes=self.NCLASS)
        # model   = SegNet(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        # model   = FCN(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        model = DefectUNet(n_channels=self.IMGSIZE[2], n_classes=self.NCLASS, bilinear=False)
        # model = ResUNet50(num_classes=self.NCLASS)

        # Pytorch官方例程中的相关网络
        # model = models.alexnet()

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
        initweight(model, type).init()

        # if loger is not None:
        #     loger.write("使用{}方法初始化网络相关权重".format(init_type))
        # else:
        #     print("使用{}方法初始化网络相关权重".format(init_type))
        logger.success("使用{}方法初始化网络相关权重".format(type))