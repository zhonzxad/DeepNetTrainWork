'''
Author: zhonzxad
Date: 2021-11-22 17:29:29
LastEditTime: 2021-11-23 13:21:44
LastEditors: zhonzxad
'''
import os
import sys

import torch
#from models.Unit.MakeVOCDataSet import MakeVOCDataSet
from models.Unit.unetdataloader import UnetDataset, dataset_collate
from models.Unit.userdataset import UserDataLoader
from torch.utils.data import DataLoader

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
os.chdir(sys.path[0])
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

class MakeLoader():
    def __init__(self, IMGSIZE=[384,384,3], CLASSNUM=2, BatchSize=2):
        self.imgsize = IMGSIZE
        self.nclass  = CLASSNUM
        self.batchsize = BatchSize
        self.tra_img = r"G:/DataBase/userdata/BXG/CutFromData-4/train/img-resize-3/"
        self.tra_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/train/label-resize-3/"
        self.val_img = r"G:/DataBase/userdata/BXG/CutFromData-4/val/img-resize-3/"
        self.val_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/val/label-resize-3/"

        # tra_img = r"G:/DataBase/userdata/BXG/CutFromData-4/train/img-resize-3/"
        # tra_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/train/label-resize-3/"
        # val_img = r"G:/DataBase/userdata/BXG/CutFromData-4/val/img-resize-3/"
        # val_leb = r"G:/DataBase/userdata/BXG/CutFromData-4/val/label-resize-3/"
        # # tra_img = r"/mnt/work/database/train/image-3"
        # # tra_leb = r"/mnt/work/database/train/label"
        # # val_img = r"/mnt/work/database/val/image-3"
        # # val_leb = r"/mnt/work/database/val/label"

        self.VOCdevkit_path = r"G:/Py_Debug/UNet-Version-master/Data/BXG/"

        with open(os.path.join(self.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
            self.train_lines = f.readlines()

        with open(os.path.join(self.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
            self.val_lines = f.readlines()
        

    def makedataUser(self):

        train_dataset = UserDataLoader(self.tra_img, self.tra_leb, image_size=self.imgsize, num_classes=self.nclass)
        val_dataset   = UserDataLoader(self.val_img, self.val_leb, image_size=self.imgsize, num_classes=self.nclass)

        # pin_memory:如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存
        # collate_fn: 将一个list的sample组成一个mini-batch的函数
        # drop_last:如果设置为True：比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
        # 另外：在Windows环境下，设置num_workers>1会及容易产生报错的现象（使用裸语句 没有被写在函数中），所以不建议在Windows下设置次参数，
        gen = DataLoader(train_dataset, shuffle=True, batch_size=self.batchsize, num_workers = 1,
                            pin_memory=True if torch.cuda.is_available() else False,
                            drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=self.batchsize, num_workers = 1,
                            pin_memory=True if torch.cuda.is_available() else False,
                            drop_last=True, collate_fn=dataset_collate)

        return gen, gen_val
    
    def makedataVoc(self):
        train_dataset   = UnetDataset(self.train_lines, self.imgsize, self.nclass, True, self.VOCdevkit_path)
        val_dataset     = UnetDataset(self.val_lines,   self.imgsize, self.nclass, False, self.VOCdevkit_path)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size=self.batchsize, 
                                    num_workers = 1, pin_memory=True,
                                    drop_last = True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = self.batchsize, 
                                    num_workers = 1, pin_memory=True,
                                    drop_last = True, collate_fn=dataset_collate)

        return gen, gen_val

    def makedata(self, backbone="User"):
        if backbone == "User":
            return self.makedataUser()
        else:
            return self.makedataVoc()
