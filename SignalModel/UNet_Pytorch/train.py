import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

from nets.unet import Unet
from nets.unet_training import weights_init
from utils_funtion.callbacks import LossHistory
from utils_funtion.dataloader import UnetDataset, unet_dataset_collate
from utils_funtion.utils_fit import fit_one_epoch
# from utils_funtion.utils_fit_transform import fit_one_epoch_transform
from utils_funtion.utils_fit_transform_uselab import fit_one_epoch_transform

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def count_param(model) -> float:
    """测试神经网络参数里
    传入 模型
    传出 参数量（像素个数，类似于分辨率的单位）
    最后结果除以10^6，最后结果的单位是M
    """
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def makedir(path:str="") -> str:
    """创建文件夹"""
    # python里的str是不可变对象，因此不存在修改一个字符串这个说法，任何对字符串的运算都会产生一个新字符串作为结果
    # 特例判断
    if path == "": return ""
    hope_path = path

    # 获取绝对路径
    # workpath = os.getcwd()
    # 或者设用下面两句话获取绝对路径
    abs_workfile_path = os.path.abspath(__file__)
    workpath, filename = os.path.split(abs_workfile_path)

    if not os.path.isabs(hope_path):
        hope_path = os.path.join(workpath, hope_path)
        # 拼合完整路径之后，如果存在文件名名称，则要去掉文件名
        # 如果传入是文件名，split会切分
        # 如果传入是路径，split不报错，返回filename为空
        hope_path, filename = os.path.split(hope_path)

    # 判断文件路径是否存在，不存在创建
    if not os.path.exists(hope_path):
        os.makedirs(hope_path)

    # 如果当前路径是 路径，自动加上下一级目录
    # 如果当前路径是 文件，拼接文件
    # 如果当前路径不代表文件，则自动加上下一级目录
    if os.path.isdir(hope_path):
        hope_path = hope_path + "/"
    if filename == "":
        hope_path = os.path.join(hope_path, filename)

    return hope_path

'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #-------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2 + 1
    #-------------------------------#
    num_classes = 2 # 2+1=3
    #-------------------------------#
    #   主干网络选择
    #   vgg、resnet50
    #-------------------------------#
    backbone    = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #--------------------------------------------------------------------------------------------------------------------------
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "model_data/unet_resnet_voc.pth"
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    input_shape = [768, 768]
    
    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 15
    Freeze_batch_size   = 2
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 30
    Unfreeze_batch_size = 2
    Unfreeze_lr         = 1e-5
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path      = 'VOCdevkit'
    VOCfile_name_source = 'Source' #'Source'
    VOCfile_name_target = 'Target'
    IsUseTransformLayer = False
    #---------------------------------------------------------------------# 
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss       = True
    #---------------------------------------------------------------------# 
    #   是否使用focal loss来防止正负样本不平衡
    #---------------------------------------------------------------------# 
    focal_loss      = False
    #---------------------------------------------------------------------# 
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #---------------------------------------------------------------------# 
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------#
    num_workers     = 0
    #------------------------------------------------------#
    #   创建记录数据tensorboard
    #------------------------------------------------------#
    tfwriter = SummaryWriter(logdir=makedir("logs/tfboard/"), comment="unet")

    model  = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone, IsUseTransformLayer=IsUseTransformLayer).train()

    # 将测试模型参数量挪到刚创建模型之后，防止后续使用CUDA报错超内存
    # paramcount_1 = count_param(model=model_train)
    # summary(model.to("cpu"), input_size=(3, input_shape[0], input_shape[1]), device='cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #  权值文件请看README，百度网盘下载
        print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        if os.path.exists(model_path):
            pretrained_dict = torch.load(model_path, map_location = device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise RuntimeError("{}不存在配置文件".format(model_path))

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.to(device)

    loss_history = LossHistory(makedir("logs/losshistory/"))
    
    # 读取数据集
    # 读取源域数据集
    with open(os.path.join(VOCdevkit_path, VOCfile_name_source, "ImageSets/Segmentation/train.txt"),"r") as f:
        source_train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, VOCfile_name_source, "ImageSets/Segmentation/val.txt"),"r") as f:
        source_val_lines = f.readlines()
    # 读取目标域数据集
    if IsUseTransformLayer: # 如果配置使用迁移网络层
        with open(os.path.join(VOCdevkit_path, VOCfile_name_target, "ImageSets/Segmentation/train.txt"),"r") as f:
            target_train_lines = f.readlines()
        with open(os.path.join(VOCdevkit_path, VOCfile_name_target, "ImageSets/Segmentation/val.txt"),"r") as f:
            target_val_lines = f.readlines()

    # 创建最优验证集损失
    best_val_loss = float("inf")

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#

    # 开始进行冻结权重训练
    if True:
        batch_size     = Freeze_batch_size
        lr             = Freeze_lr
        start_epoch    = Init_Epoch
        end_epoch      = Freeze_Epoch

        epoch_step     = len(source_train_lines) // batch_size
        epoch_step_val = len(source_val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer      = optim.Adam(model_train.parameters(), lr)
        lr_scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.96)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", \
        #                                                        verbose = True, \
        #                                                        factor=0.75, patience=1, cooldown=1, \
        #                                                        eps=1e-8)

        source_train_dataset = UnetDataset(source_train_lines, input_shape, num_classes, True,
                                           VOCdevkit_path, VOCfile_name_source, IsUseTransformLayer=False)
        source_val_dataset   = UnetDataset(source_val_lines, input_shape, num_classes, False,
                                           VOCdevkit_path, VOCfile_name_source, IsUseTransformLayer=False)
        source_gen           = DataLoader(source_train_dataset, shuffle = True, batch_size = batch_size,
                                          num_workers = num_workers, pin_memory=True,
                                          drop_last = True, collate_fn = unet_dataset_collate)
        source_gen_val       = DataLoader(source_val_dataset, shuffle = True, batch_size = batch_size,
                                          num_workers = num_workers, pin_memory=True,
                                          drop_last = True, collate_fn = unet_dataset_collate)
        # 拼接数据集tuple
        dataloads = [source_gen, source_gen_val]

        if IsUseTransformLayer: # 如果配置使用迁移网络层
            target_train_dataset = UnetDataset(target_train_lines, input_shape, num_classes, True,
                                               VOCdevkit_path, VOCfile_name_target, IsUseTransformLayer=False)
            target_val_dataset   = UnetDataset(target_val_lines, input_shape, num_classes, False,
                                               VOCdevkit_path, VOCfile_name_target, IsUseTransformLayer=False)
            target_gen           = DataLoader(target_train_dataset, shuffle = True, batch_size = batch_size,
                                              num_workers = num_workers, pin_memory=True,
                                              drop_last = True, collate_fn = unet_dataset_collate)
            target_gen_val       = DataLoader(target_val_dataset, shuffle = True, batch_size = batch_size,
                                              num_workers = num_workers, pin_memory=True,
                                              drop_last = True, collate_fn = unet_dataset_collate)
            # 拼接目标域数据集tuple
            dataloads.extend([target_gen, target_gen_val])

        # 冻结一定部分训练
        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(start_epoch, end_epoch):
            # 定义返回值为训练轮次，测试集平均损失，验证集平均损失
            if IsUseTransformLayer: # 如果配置使用迁移网络层
                ret_val = fit_one_epoch_transform(model_train, model, loss_history, optimizer, epoch,
                                    epoch_step, epoch_step_val, dataloads, end_epoch, Cuda,
                                    dice_loss, focal_loss, cls_weights, num_classes, tfwriter, best_val_loss)
            else:
                ret_val = fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                                      epoch_step, epoch_step_val, dataloads, end_epoch, Cuda,
                                      dice_loss, focal_loss, cls_weights, num_classes, tfwriter, best_val_loss)
            lr_scheduler.step()

            # 如果验证集损失下降则保存模型
            if ret_val[2] <= best_val_loss:
                best_val_loss = ret_val[2]
                save_path     = makedir("logs/pth/")
                save_filename = "Freeze_ep{:03d}-loss{:.3f}-val_loss{:.3f}_best.pth".format(
                ret_val[0], ret_val[1], ret_val[2])
                torch.save(model.state_dict(), os.path.join(save_path, save_filename))
            else:
                save_path     = makedir("logs/pth/")
                save_filename = "Freeze_ep{:03d}-loss{:.3f}-val_loss{:.3f}.pth".format(
                ret_val[0], ret_val[1], ret_val[2])
                torch.save(model.state_dict(), os.path.join(save_path, save_filename))
                print('冻结训练过程中,验证集损失没有降低，不保存参数，进入下一轮次{}'.format(epoch + 2))

    # 进入非冻结训练过程
    print('进入非解冻训练阶段')
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = len(source_train_lines) // batch_size
        epoch_step_val  = len(source_val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.96)

        source_train_dataset   = UnetDataset(source_train_lines, input_shape, num_classes, True,
                                             VOCdevkit_path, VOCfile_name_source, IsUseTransformLayer=False)
        source_val_dataset     = UnetDataset(source_val_lines, input_shape, num_classes, False,
                                             VOCdevkit_path, VOCfile_name_source, IsUseTransformLayer=False)
        source_gen             = DataLoader(source_train_dataset, shuffle = True, batch_size = batch_size,
                                            num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate)
        source_gen_val         = DataLoader(source_val_dataset, shuffle = True, batch_size = batch_size,
                                            num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate)
        # 拼接数据集tuple
        dataloads = [source_gen, source_gen_val]

        if IsUseTransformLayer: # 如果配置使用迁移网络层
            target_train_dataset = UnetDataset(target_train_lines, input_shape, num_classes, True,
                                               VOCdevkit_path, VOCfile_name_target, IsUseTransformLayer=False)
            target_val_dataset   = UnetDataset(target_val_lines, input_shape, num_classes, False,
                                               VOCdevkit_path, VOCfile_name_target, IsUseTransformLayer=False)
            target_gen           = DataLoader(target_train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                              drop_last = True, collate_fn = unet_dataset_collate)
            target_gen_val       = DataLoader(target_val_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                              drop_last = True, collate_fn = unet_dataset_collate)
            # 拼接目标域数据集tuple
            dataloads.extend([target_gen, target_gen_val])

        # 解冻网络参数，参与权重训练
        if Freeze_Train:
            model.unfreeze_backbone()

        for epoch in range(start_epoch,end_epoch):
            # 定义返回值为训练轮次，测试集平均损失，验证集平均损失
            if IsUseTransformLayer: # 如果配置使用迁移网络层
                ret_val = fit_one_epoch_transform(model_train, model, loss_history, optimizer, epoch,
                                                  epoch_step, epoch_step_val, dataloads, end_epoch, Cuda,
                                                  dice_loss, focal_loss, cls_weights, num_classes, tfwriter, best_val_loss)
            else:
                ret_val = fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                                        epoch_step, epoch_step_val, dataloads, end_epoch, Cuda,
                                        dice_loss, focal_loss, cls_weights, num_classes, tfwriter, best_val_loss)
            lr_scheduler.step()

            # 如果验证集损失下降则保存模型
            if ret_val[2] <= best_val_loss:
                best_val_loss = ret_val[2]
                save_path     = makedir("logs/pth/")
                save_filename = "UNFreeze_ep{:03d}-loss{:.3f}-val_loss{:.3f}_best.pth".format(
                ret_val[0], ret_val[1], ret_val[2])
                torch.save(model.state_dict(), os.path.join(save_path, save_filename))
            else:
                save_path     = makedir("logs/pth/")
                save_filename = "UNFreeze_ep{:03d}-loss{:.3f}-val_loss{:.3f}.pth".format(
                ret_val[0], ret_val[1], ret_val[2])
                torch.save(model.state_dict(), os.path.join(save_path, save_filename))
                print('非冻结训练过程中,验证集损失没有降低，进入下一轮次{}'.format(epoch + 2))


