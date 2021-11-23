# -*- coding: UTF-8 -*- 
'''
Author: zhonzxad
Date: 2021-10-26 10:34:44
LastEditTime: 2021-11-23 10:14:00
LastEditors: zhonzxad
'''
import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.Unit.getmodel import GetModel
from models.Unit.writelog import WriteLog

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
os.chdir(sys.path[0])

# 创建全局写日志对象
global writer

# 测试方法
def test(model, image):
    model.eval()

    image = [np.array(image) / 255]
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():
        image = torch.from_numpy(image).type(torch.FloatTensor)
        if GPU:
            image = image.to(this_device)

    # 模型预测
    retimg = model(image)

    # 最终返回结果已去掉bitch维度
    return retimg[0] 

# 根据网络输出结果创建结果图片
def CreatSeqImg(img, num_classes=2):
    img = img.permute(1, 2, 0)
    # 取出每一个像素点的种类,  argmax返回最大值的索引
    ret = torch.softmax(img, dim=-1).cpu().detach().numpy().argmax(axis=-1)
    # ret = torch.softmax(img, dim=-1).cpu().numpy()

    #------------------------------------------------#
    #   创建一副新图，并根据每个像素点的种类赋予颜色
    #------------------------------------------------#
    colors = [(0, 0, 0), (128, 0, 0)]
    seg_img = np.zeros((np.shape(ret)[0], np.shape(ret)[1], 3))
    for c in range(num_classes):
        seg_img[:,:,0] += ((ret[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((ret[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,2] += ((ret[:,: ] == c )*( colors[c][0] )).astype('uint8')
    
    return Image.fromarray(np.uint8(seg_img))


def CutFullImg(image):
    '''
    在对原图和标签图同时进行裁剪，裁剪为四分
    img: 原图图片
    '''
    len = image.size[0] 
    wid = image.size[1]
    wid_mid = int(wid // 2)
    len_mid = int(len // 2)
    # dst = img[200:600, 0:300]   # 裁剪坐标为[y0:y1, x0:x1]

    cutimagelist = list()
    # cutimagelist.append(image[0:wid_mid, 0:len_mid])      # 左上
    # cutimagelist.append(image[0:wid_mid, len_mid:len])    # 右下
    # cutimagelist.append(image[wid_mid:wid, 0:len_mid])    # 左下
    # cutimagelist.append(image[wid_mid:wid, len_mid:len])  # 右下
    cutimagelist.append(image.crop( (      0,       0, len_mid, wid_mid) )) # 左上
    cutimagelist.append(image.crop( (len_mid,       0,     len, wid_mid) )) # 左下
    cutimagelist.append(image.crop( (      0, wid_mid, len_mid,     wid) )) # 右上
    cutimagelist.append(image.crop( (len_mid, wid_mid,     len,     wid) )) # 右下

    return cutimagelist

def ImgMerge(cutimagelist):
    """
    四个图像合并，要求按照CutFullImg切割的图像
    cutimagelist：原始图切分后数组
    """
    for i in range(4):
        if type(cutimagelist[i]) is np.ndarray:
            cutimagelist[i] = Image.fromarray(np.uint8(cutimagelist[i]))
        elif type(cutimagelist[i]) is Image.Image:
            pass
        else:
            raise TypeError("合并数据类型不正确")

    len = cutimagelist[0].size[0]
    wid = cutimagelist[0].size[1]

    new_image = Image.new(cutimagelist[0].mode, (len * 2, wid * 2))#, color='white')
    # cutimagelist[i] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    new_image.paste(cutimagelist[0], (  0,   0) )       # 左上
    new_image.paste(cutimagelist[1], (len,   0) )       # 左下
    new_image.paste(cutimagelist[2], (  0, wid) )       # 右上
    new_image.paste(cutimagelist[3], (len, wid) )       # 右下

    # leftimg = np.concatenate((cutimagelist[0], cutimagelist[2]), axis=0)  # axis=0 按垂直方向，axis=1 按水平方向
    # rightimg = np.concatenate((cutimagelist[1], cutimagelist[3]), axis=0)  # axis=0 按垂直方向，axis=1 按水平方向
    # img = np.concatenate((leftimg,rightimg), axis=1)

    return new_image

if __name__ == '__main__':
    UseGPU      = False
    SEED        = 2021           # 设置随机种子
    # 为CPU设定随机种子使结果可靠，就是每个人的随机结果都尽可能保持一致
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    this_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU = True if this_device.type == "cuda" and True else False

    # 加载日志对象
    writer = WriteLog(writerpath=r'./log/')

    CLASSNUM = 2
    IMGSIZE  = [384, 384, 3]

    # 加载模型
    model = GetModel([IMGSIZE, CLASSNUM], writer).Createmodel(is_train=False)
    # UNet(input_channels=IMGSIZE[2], num_class=CLASSNUM)
    if GPU:
        model = model.to(this_device)
    writer.write("网络创建完毕")
    
    path = "./savepoint/model_data/UNet_2Class_NewLoss_1.pth"            # 本机权重
    # path = "./savepoint/model_data/UNET-2Class-NewData-checkpoint.pth"      # 服务器权重
    if os.path.isfile(path):
        model_data = torch.load(path, map_location="cuda:0" if GPU else "cpu")
        model = model_data['model']
        writer.write("加载参数完成")
    else:
        raise RuntimeError

    image = Image.open(r"C:/Users/zxuan/Desktop/color_0022.jpg")
    print(image.size)
    # img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    # cv2.imshow("Ori", img)

    CutImgList = CutFullImg(image)

    RetImgList = []
    for i in range(len(CutImgList)):
        ret = test(model, CutImgList[i])
        RetImgList.append(ret)

    for i in range(len(RetImgList)):
        RetImgList[i] = CreatSeqImg(RetImgList[i]).copy()

    RetImg = ImgMerge(RetImgList)

    savepath = r"C:/Users/zxuan/Desktop/UNet-2Class-NewLoss-zx1-0001.jpg"
    RetImg.save(savepath)
    print("预测完成，预测结果保存在{}".format(savepath))

    # img = cv2.cvtColor(np.asarray(RetImg),cv2.COLOR_RGB2BGR)
    # cv2.imshow("Ret", img)
    # cv2.imwrite(r"C:/Users/zxuan/Desktop/ret.jpg", ret)
    # while True:
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
            # break

