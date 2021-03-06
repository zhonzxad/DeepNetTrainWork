'''
Author: zhonzxad
Date: 2021-11-22 17:25:31
LastEditTime: 2021-12-01 09:59:15
LastEditors: zhonzxad
'''
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset



def cvtColorRGB(image):
    # 将图像转换成RGB图像，防止灰度图在预测时报错。
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size):
    # 对输入图像进行resize
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
# 图像均一化
def preprocess_input(image):
    image /= 255.0
    return image

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shapewh, num_classes, train, dataset_path, VOC_FileName):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shapewh
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.VOCFileName        = VOC_FileName

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, 
                                                    self.VOCFileName + "/JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, 
                                                    self.VOCFileName + "/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])

        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        # 转化成one_hot的形式
        seg_label  = np.eye(self.num_classes)[png.copy().reshape([-1])]
        seg_label  = seg_label.reshape((int(self.input_shape[0]), int(self.input_shape[1]),
                                            self.num_classes))
        # seg_label = png.copy().astype(np.int64)
        # seg_label = torch.from_numpy(seg_label)
        # seg_label = F.one_hot(seg_label, num_classes=self.num_classes)
        # seg_label = seg_label.numpy()

        # png = png.transpose(2, 0, 1)
        # seg_label = seg_label.transpose(2, 0, 1)

        return jpg, png, seg_label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = cvtColorRGB(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # resize image
        rand_jit1 = self.rand(1 - jitter, 1 + jitter)
        rand_jit2 = self.rand(1 - jitter, 1 + jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)

        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data, label
