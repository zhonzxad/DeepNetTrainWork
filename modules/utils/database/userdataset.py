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
from torch.utils.data.dataset import Dataset

transforms_train = transforms.Compose([
                    transforms.ToPILImage(),             
                    transforms.ToTensor(),                                  # 将数据转换成Tensor型
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])     # 标准化

class maketestonehot():
    def __init__(self):
        self.colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

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

    def mask2onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector
        """
        semantic_map = [mask == i for i in range(num_classes)]
        return np.array(semantic_map).astype(np.uint8)

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

    def png_to_onehot(self, png):
        """
        Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one
        hot encoding vector, C is usually 1 or 3, and K is the number of class.
        """
        colormap = [[0, 0, 0], [128, 0, 0],]
        semantic_map = []
        for colour in colormap:
            equality = np.equal(png, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map

    def __getitem__(self, index):
        # G:/Py_Debug/pspnet-pytorch-master/VOCdevkit/VOC2007/ImageSets/
        # 从文件中读取图像
        imgfilepath = self.imgpath + self.imgpath_list[index]  # 组合原始图片路径
        filetitle = os.path.split(imgfilepath)[1]
        shotname, extension = os.path.splitext(filetitle)
        labelfilepath = self.labelpath + shotname + ".png"   # 取出对应标签图片路径

        # 判断找图是否正确,==0说明list列表里没有对应文件
        if self.labelpath_list.count(shotname + ".png") == 0:
            raise RuntimeError("未找到原图对应的标签文件")

        jpg = Image.open(imgfilepath).convert("L")    # 统一转为三通道格式读取
        png = Image.open(labelfilepath).convert('L')    # 统一转为单通道格式读取

        # 20220224 不需要做出resize，这里的resize是从配置文件中设置的
        # 现在的方式是由文件原始大小决定，此时应确保原始文件的正确性
        # Image.BILINEAR 双线性插值
        # jpg = jpg.resize((self.image_size[0], self.image_size[1]), Image.BILINEAR)
        # png = png.resize((self.image_size[0], self.image_size[1]), Image.BILINEAR)

        # 处理jpg格式变换
        jpg = np.transpose(np.array(jpg).reshape(self.image_size[0], self.image_size[1], self.image_size[2]), [2, 0, 1])
        # jpg = np.array(jpg)
        # jpg = torch.from_numpy(jpg)

        # 处理png格式变化
        # 产生数组
        png = np.array(png)
        # png[png >= self.num_classes] = self.num_classes
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
        # seg_labels = seg_labels.astype(np.int64)
        # seg_labels = torch.from_numpy(seg_labels)
        # seg_labels = F.one_hot(seg_labels, num_classes=self.num_classes)
        # seg_labels = seg_labels.numpy()
        # seg_labels = seg_labels.permute(2, 0, 1)

        seg_labels = seg_labels.reshape(png.shape[0], png.shape[1], 1)
        seg_labels = self.png_to_onehot(seg_labels)

        # seg_labels = np.transpose(np.array(jpg), [2,0,1])
        # seg_labels = np.transpose(seg_labels.numpy(), [2,0,1])
        # seg_labels = torch.from_numpy(seg_labels)
        # seg_labels = self.makeonehotlab(png)
        # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3).contiguous()

        # seg_labels = torch.nn.functional.one_hot(t, num_classes=self.num_classes)

        # seg_labels = seg_labels.reshape(self.image_size[0], self.image_size[1], 1)


        return jpg, png, seg_labels



