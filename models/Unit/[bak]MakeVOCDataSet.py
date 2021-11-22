import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


# 使用填充以不变的长宽比调整图像大小
# 不失真的resize
def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# 获取随机选择出来的图
def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w, c = input_shape
        # resize image
        rand_jit1 = rand(1-jitter, 1+jitter)
        rand_jit2 = rand(1-jitter, 1+jitter)
        new_ar = w/h * rand_jit1 / rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        
        # Image插值时按照BICUBIC双三次插值
        image = image.resize((nw, nh), Image.BICUBIC)
        # NEAREST最邻近插值
        label = label.resize((nw, nh), Image.NEAREST)
        # 转为灰度图像
        label = label.convert("L")
        
        # 是否反转图像
        flip = rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image？
        # 给图像加遮罩
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        # 扭曲图像
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        return image_data, label

class MakeVOCDataSet(Dataset):
    def __init__(self, dataroot, isranddata):
        super(MakeVOCDataSet, self).__init__()
        self.dataroot = dataroot
        self.isranddata = isranddata
        self.img_data = []
        self.lab_data = []

        for root, _, files in os.walk(self.dataroot.join("img")):
            for file in files:
                if os.path.splitext(file)[1] == '.jpeg':
                    self.img_data.append(os.path.join(root, file))

        for root, _, files in os.walk(self.dataroot.join("lab")):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    self.lab_data.append(os.path.join(root, file))

    def getlist(self):
        return self.img_data, self.lab_data

    def getdata(self, index):
        assert (index <= self.getlen())
        return self.img_data[index], self.lab_data[index]

    def getlen(self):
        assert (self.train_data == self.val_data)

        return len(self.train_data)

    def __len__(self):
        assert (self.train_data == self.val_data)

        return len(self.train_data)

    def __getitem__(self, index):
        jpg, png = self.getdata(index)
        if self.isranddata:
            jpg, png = get_random_data(jpg, png,
                                        (int(self.image_size[0]), int(self.image_size[1])))
        else:
            jpg, png = letterbox_image(jpg, png, 
                                        (int(self.image_size[1]), int(self.image_size[0])))

        # 从文件中读取图像
        # 产生数组
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        # 转化成one_hot的形式
        seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[0]), int(self.image_size[1]), self.num_classes+1))

        jpg = np.transpose(np.array(jpg), [2,0,1])/255

        return jpg, png, seg_labels

# DataLoader中collate_fn使用
def dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    images      = np.array(images)
    pngs        = np.array(pngs)
    seg_labels  = np.array(seg_labels)
    return images, pngs, seg_labels


