'''
Author: zhonzxad
Date: 2021-10-21 22:26:30
LastEditTime: 2021-11-29 19:34:36
LastEditors: zhonzxad
'''
import math
import os
from random import shuffle

import cv2
import numpy as np
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

transforms_train = transforms.Compose([
                    transforms.ToPILImage(),             
                    transforms.ToTensor(),                                  # 将数据转换成Tensor型
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])     # 标准化

# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []

    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    # 产生数组
    images     = np.array(images)
    pngs       = np.array(pngs)
    seg_labels = np.array(seg_labels)

    return images, pngs, seg_labels


class UserDataLoaderTrans(Dataset):
    def __init__(self, imgpath, image_size, num_classes):
        super(UserDataLoaderTrans, self).__init__()

        self.imgpath = imgpath
        self.imgpath_list = list(os.listdir(self.imgpath))

        self.train_count   = len(self.imgpath_list)
        self.image_size    = image_size
        self.num_classes   = num_classes

    def __len__(self):
        return self.train_count

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def makeonehotlab(self, lab):
        '''
        将一维列表转换为独热编码
        '''

        self.batch_size = 2
        lab = torch.from_numpy(np.array(lab))
        lab = lab.view(-1)
        label = lab.resize_(self.batch_size, 1).long()
        m_zeros = torch.zeros(self.batch_size, self.num_classes)
        # 从value中取值,然后根据dim和index给相应位置赋值
        onehot = m_zeros.scatter_(1, label, 1.)  # (dim, index, value)
        
        return onehot

    def __getitem__(self, index):
        # 从文件中读取图像
        imgfilepath = self.imgpath + self.imgpath_list[index]  # 组合原始图片路径

        jpg = Image.open(imgfilepath).convert("RGB")    # 统一转为三通道格式读取

        jpg = jpg.resize((self.image_size[0], self.image_size[1]), Image.BILINEAR)

        # 处理jpg格式变换
        jpg = np.transpose(np.array(jpg), [2, 0, 1])
        # jpg = np.array(jpg)
        # jpg = torch.from_numpy(jpg)

        return jpg


