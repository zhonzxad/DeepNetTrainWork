'''
Author: zhonzxad
Date: 2021-11-22 17:03:36
LastEditTime: 2021-11-30 10:00:45
LastEditors: zhonzxad
'''
import os
import random
import sys

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
os.chdir(sys.path[0])

#----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#----------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.8
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = r'//Data/BXG/'

if __name__ == "__main__":
    random.seed(2021)
    print("Generate txt in ImageSets.")
    imgfilepath     = os.path.join(VOCdevkit_path, 'Signal-VOC/JPEGImages')
    segfilepath     = os.path.join(VOCdevkit_path, 'Signal-VOC/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'Signal-VOC/ImageSets/Segmentation')
    
    img_list = os.listdir(imgfilepath)
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            name, end = seg.split(".")
            imagename = name + ".jpg"
            count = img_list.count(imagename)
            if count is 1:
                total_seg.append(seg)

    num      = len(total_seg)  
    list     = range(num)  
    tv       = int(num * trainval_percent)  
    tr       = int(tv * train_percent)  
    trainval = random.sample(list, tv)  
    train    = random.sample(trainval, tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath, 'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath, 'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath, 'val.txt'), 'w')  
    
    for i  in list:  
        name=total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")
