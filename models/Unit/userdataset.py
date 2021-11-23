'''
Author: zhonzxad
Date: 2021-10-21 22:26:30
LastEditTime: 2021-11-23 22:02:34
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


class UserDataLoader(Dataset):
    def __init__(self, imgpath, labelpath, image_size, num_classes):
        super(UserDataLoader, self).__init__()

        self.imgpath = imgpath
        self.imgpath_list = list(os.listdir(self.imgpath))

        self.labelpath = labelpath
        self.labelpath_list = list(os.listdir(self.labelpath))
        
        assert(len(self.imgpath_list) == len(self.labelpath_list))

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
        # G:/Py_Debug/pspnet-pytorch-master/VOCdevkit/VOC2007/ImageSets/
        # 从文件中读取图像
        imgfilepath = self.imgpath + "/" + self.imgpath_list[index]  # 组合原始图片路径
        filetitle = os.path.split(imgfilepath)[1]
        shotname, extension = os.path.splitext(filetitle)
        labelfilepath = self.labelpath + "/" + shotname + ".png"   # 取出对应标签图片路径

        jpg = Image.open(imgfilepath).convert("RGB")    # 统一转为三通道格式读取
        png = Image.open(labelfilepath).convert('L')    # 统一转为单通道格式读取

        jpg = jpg.resize((self.image_size[0], self.image_size[1]), Image.BILINEAR)
        png = png.resize((self.image_size[0], self.image_size[1]), Image.BILINEAR)

        # 处理jpg格式变换
        jpg = np.transpose(np.array(jpg), [2, 0, 1])
        # jpg = np.array(jpg)
        # jpg = torch.from_numpy(jpg)

        # 处理png格式变化
        # 产生数组
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = png.copy()
        # png = np.transpose(np.array(png), [2,0,1])
        # png = png.unsqueeze(dim=-1)
        # png = torch.from_numpy(png)
        # png = png.unsqueeze(dim=0)
        # png = torch.cat([png, png.clone()],dim=0)
        
        # 转化成one_hot的形式
        # seg_labels = np.eye(self.num_classes)[np.array(seg_labels).reshape([-1])]
        # seg_labels = seg_labels.reshape(int(self.image_size[0]), int(self.image_size[1]), self.num_classes)
        # seg_labels = torch.from_numpy(seg_labels).permute(2, 0, 1)
        
        # seg_labels 在创建的时候被赋值为int64
        seg_labels = seg_labels.astype(np.int64)
        seg_labels = torch.from_numpy(seg_labels)
        seg_labels = F.one_hot(seg_labels, num_classes=self.num_classes)
        seg_labels = seg_labels.numpy()
        # seg_labels = seg_labels.permute(2, 0, 1)

        # seg_labels = np.transpose(np.array(jpg), [2,0,1])
        # seg_labels = np.transpose(seg_labels.numpy(), [2,0,1])
        # seg_labels = torch.from_numpy(seg_labels)
        # seg_labels = self.makeonehotlab(png)
        # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3).contiguous()

        # seg_labels = torch.nn.functional.one_hot(t, num_classes=self.num_classes)

        # seg_labels = seg_labels.reshape(self.image_size[0], self.image_size[1], 1)


        return jpg, png, seg_labels

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


