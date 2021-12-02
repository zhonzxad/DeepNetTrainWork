'''
Author: zhonzxad
Date: 2021-11-23 09:49:29
LastEditTime: 2021-12-02 21:27:34
LastEditors: zhonzxad
'''
import argparse

import torch

from .FCN.fcn import FCN
from .ResNet.ResNet import GetResNet
from .ResNet.resnet18 import RestNet18
from .SegNet.SegNet import SegNet
from .SmaAtUNer.SmaAt_UNet import SmaAtUNet
from .UNet.UNet import UNet
from .UNet.UNet_2Plus import UNet_2Plus
from .UNet.UNet_3Plus import UNet_3Plus
from .UNet.UNetBili import UNetVGG16

# import os
# import sys

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

class GetModel():
    def __init__(self, args, loger):
        if type(args) == argparse.Namespace:
            self.args    = args
            self.IMGSIZE = self.args.IMGSIZE
            self.NCLASS  = self.args.nclass
        else:
            self.IMGSIZE = args[0]
            self.NCLASS  = args[1]
            
        self.writer = loger

    def weights_init(self, net, loger, init_type='normal', init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

        if loger is not None:
            loger.write("使用{}方法初始化网络相关权重".format(init_type))
        else:
            print("使用{}方法初始化网络相关权重".format(init_type))

    def Createmodel(self, is_train=True):
        # 加载模型
        # model = UNet(input_channels=self.IMGSIZE[2], num_class=self.NCLASS)
        # model = UNetVGG16(num_classes=CLASSNUM, in_channels=IMGSIZE[2])
        # model = UNet_2Plus(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model = RestNet18(in_channels=IMGSIZE[2], n_classes=CLASSNUM)
        # model   = SegNet(input_channels=IMGSIZE[2], num_class=CLASSNUM)
        # model   = FCN(input_channels=IMGSIZE[2], num_class=CLASSNUM)
        model = SmaAtUNet(n_channels=self.IMGSIZE[2], n_classes=self.NCLASS)

        # 初始化网络相关权重
        if is_train:
            self.weights_init(model, loger=self.writer, init_type="kaiming")
            model = model.train()
        else:
            model = model.eval()

        return model
