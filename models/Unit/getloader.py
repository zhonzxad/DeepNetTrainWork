'''
Author: zhonzxad
Date: 2021-11-22 17:29:29
LastEditTime: 2021-12-01 09:50:56
LastEditors: zhonzxad
'''
import os
import sys
import numpy as np
# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
sys.path.append("")

import torch
from torch.utils.data import DataLoader

from models.Unit.makedata.userdataset import UserDataLoader
from models.Unit.makedata.userdataset_trans import UserDataLoaderTrans
from models.Unit.makedata.unetdataloader import UnetDataset

"""
train_dataser = MakeVOCDataSet.MakeVOCDataSet(args.root_path.join("train"))
val_dataset   = MakeVOCDataSet.MakeVOCDataSet(args.root_path.join("val"))

# pin_memory:如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
# collate_fn: 将一个list的sample组成一个mini-batch的函数
# drop_last:如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
train_loader = torch.utils.data.DataLoader(train_dataser, shuffle=True, batch_size=args.batch_size, num_workers=1,
                    pin_memory=True if torch.cuda.is_available() else False,
                    drop_last=True, collate_fn=MakeVOCDataSet.dataset_collate)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False,
                    drop_last=True, collate_fn=MakeVOCDataSet.dataset_collate)
"""

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

class GetLoader():
    def __init__(self, IMGSIZE=[384,384,3], CLASSNUM=2, BatchSize=2, num_workers=1):
        self.imgsize = IMGSIZE
        self.nclass  = CLASSNUM
        self.batchsize = BatchSize
        self.num_workers = num_workers

        self.tra_img = r"G:/DataBase/userdata/BXG/CutFromData-4/train/img-resize-3/"
        self.tra_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/train/label-resize-3/"
        self.val_img = r"G:/DataBase/userdata/BXG/CutFromData-4/val/img-resize-3/"
        self.val_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/val/label-resize-3/"
        # self.tra_img = r"G:/DataBase/userdata/BXG/CutFromData-4/train/img-resize-3/"
        # self.tra_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/train/label-resize-3/"
        # self.val_img = r"G:/DataBase/userdata/BXG/CutFromData-4/val/img-resize-3/"
        # self.val_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/val/label-resize-3/"
        # self.tra_img = r"/mnt/work/database/BXG/train/img-resize-3/"
        # self.tra_leb = r"/mnt/work/database/BXG/train/label-resize-3/"
        # self.val_img = r"/mnt/work/database/BXG/val/img-resize-3/"
        # self.val_leb = r"/mnt/work/database/BXG/val/label-resize-3/"
        
        self.VOCFileName    = "Signal-VOC"
        self.VOCdevkit_path = r"G:/Py_Debug/UNet-Version-master/Data/BXG/"

    def makedataUser(self):

        train_dataset = UserDataLoader(self.tra_img, self.tra_leb,
                                        image_size=self.imgsize, num_classes=self.nclass)
        val_dataset   = UserDataLoader(self.val_img, self.val_leb,
                                        image_size=self.imgsize, num_classes=self.nclass)

        # pin_memory:如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
        # collate_fn: 将一个list的sample组成一个mini-batch的函数
        # drop_last:如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
        # 另外：在Windows环境下，设置num_workers>1会及容易产生报错的现象（使用裸语句 没有被写在函数中），所以不建议在Windows下设置次参数，
        gen = DataLoader(train_dataset, shuffle=True, batch_size=self.batchsize, num_workers=self.num_workers,
                            pin_memory=True if torch.cuda.is_available() else False,
                            drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=self.batchsize, num_workers=self.num_workers,
                            pin_memory=True if torch.cuda.is_available() else False,
                            drop_last=True, collate_fn=dataset_collate)

        return gen, gen_val
    
    def makedataVoc(self):

        with open(os.path.join(self.VOCdevkit_path, self.VOCFileName + "/ImageSets/Segmentation/train.txt"), "r") as f:
            self.train_lines = f.readlines()

        with open(os.path.join(self.VOCdevkit_path, self.VOCFileName + "/ImageSets/Segmentation/val.txt"), "r") as f:
            self.val_lines = f.readlines()

        train_dataset = UnetDataset(self.train_lines, self.imgsize, self.nclass, 
                                      True, self.VOCdevkit_path, self.VOCFileName)
        val_dataset   = UnetDataset(self.val_lines,   self.imgsize, self.nclass, 
                                      False, self.VOCdevkit_path, self.VOCFileName)

        gen           = DataLoader(train_dataset, shuffle = True, batch_size=self.batchsize, 
                                  num_workers=self.num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=dataset_collate)
        gen_val       = DataLoader(val_dataset  , shuffle = True, batch_size = self.batchsize, 
                                    num_workers=self.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)

        return gen, gen_val

    def makedataUser_Targer(self):
        train_dataset = UserDataLoaderTrans(self.tra_img,
                                       image_size=self.imgsize, num_classes=self.nclass)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=self.batchsize, num_workers=self.num_workers,
                         pin_memory=True if torch.cuda.is_available() else False,
                         drop_last=True, collate_fn=dataset_collate)

        return gen

    def makedata(self):
        """
        源域数据集/source
        """
        return self.makedataVoc()

    def makedataTarget(self):
        """
        目标域数据集/target
        """
        return self.makedataUser_Targer()
