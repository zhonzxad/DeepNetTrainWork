'''
Author: zhonzxad
Date: 2021-11-23 09:49:29
LastEditTime: 2021-11-23 14:18:58
LastEditors: zhonzxad
'''
import argparse
import sys

sys.path.append("..")

from ..Net.FCN.fcn import FCN
from ..Net.SegNet.SegNet import SegNet
from ..Net.UNet.resnet18 import RestNet18
from ..Net.UNet.UNet import UNet
from ..Net.UNet.UNet_2Plus import UNet_2Plus
from ..Net.UNet.UNetBili import UNetVGG16
from ..Unit.pytorchtools import weights_init


class  GetModel():
    def __init__(self, args, loger):
        if type(args) == argparse.Namespace:
            self.args    = args
            self.IMGSIZE = self.args.IMGSIZE
            self.NCLASS  = self.args.nclass
        else:
            self.IMGSIZE = args[0]
            self.NCLASS  = args[1]
            
        self.writer = loger

    def Createmodel(self, is_train=True):
        # 加载模型
        model = UNet(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        # model = UNetVGG16(num_classes=CLASSNUM, in_channels=IMGSIZE[2])
        # model = UNet_2Plus(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model = RestNet18(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model   = SegNet(input_channels=IMGSIZE[2], num_class=CLASSNUM)
        # model   = FCN(input_channels=IMGSIZE[2], num_class=CLASSNUM)

        # 初始化网络相关权重
        if is_train:
            weights_init(model, loger=self.writer)
            model = model.train()
        else:
            model = model.eval()

        return model
