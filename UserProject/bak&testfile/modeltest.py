# -*- coding: utf-8 -*-
# @Time     : 2021/12/7 上午 09:49
# @Author   : zhonzxad
# @File     : modeltest.py
import math
import os
from random import shuffle

import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from torch import tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class maketestonehot():
    def __init__(self):
        self.colormap = [[0, 0, 0], [128, 0, 0],]

        self.classes = ['background', 'crack', ]

    def label_to_onehot(self, label):
        """
        Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one
        hot encoding vector, C is usually 1 or 3, and K is the number of class.
        """
        semantic_map = []
        for colour in self.colormap:
            equality = np.equal(label, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map

    def label_to_onehot_1(self, label, num_classes):
        # 转化成one_hot的形式
        h, w, c = label.shape
        seg_labels = np.eye(num_classes)[np.array(label).reshape([-1])]
        seg_labels = seg_labels.reshape(int(h), int(w), num_classes).astype(np.float32)
        # seg_labels = torch.from_numpy(seg_labels).permute(2, 0, 1)
        return seg_labels

    def label_to_onehot_2(self, label, num_classes):
        # seg_labels 在创建的时候被赋值为int64
        seg_labels = label.astype(np.int64)
        seg_labels = torch.from_numpy(seg_labels)
        seg_labels = F.one_hot(seg_labels, num_classes=num_classes)
        seg_labels = seg_labels.numpy()
        # seg_labels = seg_labels.permute(2, 0, 1)
        return seg_labels

    def mask2onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector
        """
        semantic_map = [mask == i for i in range(num_classes)]
        return np.array(semantic_map).astype(np.uint8)

if __name__ == '__main__':

    def procedure():
        for i in range(10000):
            pass

    # time.time
    t0 = time.time()
    procedure()
    print (time.time() - t0)

    png = Image.open(r"G:\Py_Debug\unet-pytorch-main\datasets\SegmentationClass\1.png")    # 统一转为单通道格式读取


    # png[png >= 2] = 2

    makere = maketestonehot()

    png_0 = png.copy()
    png_0 = np.array(png_0).reshape(png.size[0], png.size[1], 1)
    ret1 = makere.label_to_onehot(png_0)

    png_1 = png.copy()
    png_1 = np.array(png_1).reshape(png.size[0], png.size[1], 1)
    png_1[png_1 >= 2] = 2
    ret3 = makere.label_to_onehot_1(png_1, 2)

    print(np.array_equal(ret1, ret3))

    pass
