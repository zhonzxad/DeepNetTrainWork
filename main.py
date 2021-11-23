# -*- coding: UTF-8 -*- 
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from loss.bceloss import BCELoss2d
from loss.celoss import CELOSS, CELoss2d
from loss.diceloss import Dice_Loss, DiceLoss
from loss.fscore import f_score
from loss.iouloss import bbox_overlaps_ciou
from models.Unit.getmodel import GetModel
from models.Unit.getoptim import CreateOptim
from models.Unit.makeloader import MakeLoader
from models.Unit.pytorchtools import EarlyStopping
from models.Unit.writelog import WriteLog

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
os.chdir(sys.path[0])

# 创建全局对象
global writer
global tfwriter
global this_device

# 获取学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 定义训练每一个epoch的步骤
def fit_one_epoch(model, epoch, dataloaders, optimizer, scheduler):

    total_ce_loss   = 0
    total_bce_loss  = 0
    total_dice_loss = 0
    total_f_score   = 0
    total_loss      = 0

    model_train = model.train()

    tqdmbar = tqdm(dataloaders)
    for batch_idx, batch in enumerate(tqdmbar):

        img, png, label = batch
        # print("\nNo in RangeNet img shape is {} || png shape is {}".format(img.shape, png.shape))

        with torch.no_grad():
            # img = torch.autograd.Va   riable(torch.from_numpy(img).type(torch.FloatTensor))
            # png = torch.autograd.Variable(torch.from_numpy(png).type(torch.FloatTensor)).long()
            # seg_labels = torch.autograd.Variable(torch.from_numpy(seg_labels).type(torch.FloatTensor))
            img    = torch.from_numpy(img).type(torch.FloatTensor)
            png    = torch.from_numpy(png).type(torch.FloatTensor)
            label  = torch.from_numpy(label).type(torch.FloatTensor)
            # img = torch.autograd.Variable(img).type(torch.FloatTensor)
            # png = torch.autograd.Variable(png).type(torch.FloatTensor)
            # seg_labels = torch.autograd.Variable(seg_labels).type(torch.FloatTensor)
            # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3)
            # writer.write("\n img shape is {} || png shape is {} || seg_labels shape is {}".format(img.shape, png.shape, seg_labels.shape))

            if this_device.type == "cuda":
                img = img.to(this_device)
                png = png.to(this_device)
                label = label.to(this_device)

        # 所有梯度为0
        optimizer.zero_grad()

        # 网络计算
        output = model_train(img)

        # 计算损失
        # print("\n output shape is {} || png shape is {}".format(output.shape, png.shape))
        # ce_loss   = CELOSS(output, png)
        ce_loss   = CELoss2d()(output, png)
        bce_loss  = BCELoss2d()(output, label)
        dice_loss = DiceLoss()(output, label)
        loss = ce_loss + dice_loss
        
        with torch.no_grad():
            _f_score = f_score(output, label)

        # 误差反向传播
        loss.backward()
        # 优化梯度
        optimizer.step()

        total_ce_loss   += ce_loss.item()
        total_bce_loss  += bce_loss.item()
        total_dice_loss += dice_loss.item()
        total_f_score   += _f_score.item()
        total_loss      += loss.item()

        total_ce_loss   /= (batch_idx + 1)
        total_bce_loss  /= (batch_idx + 1)
        total_dice_loss /= (batch_idx + 1)
        total_f_score   /= (batch_idx + 1)
        total_loss      /= (batch_idx + 1)

        # 写tensorboard
        tags = ["train_loss", "CEloss", "BCEloss", "Diceloss", "f_score", "lr", "accuracy"]
        if tfwriter != None:
            tfwriter.add_scalar(tags[0],        total_loss)#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[1],     total_ce_loss)#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[2],    total_bce_loss)#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[3],         dice_loss)#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[4],     total_f_score)#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[5], get_lr(optimizer))#, epoch*(batch_idx + 1))

        #设置进度条左边显示的信息
        tqdmbar.set_description("Epoch in Range")
        #设置进度条右边显示的信息
        tqdmbar.set_postfix(Loss=("{:5f}".format(total_loss)),
                            CEloss=("{:5f}".format(total_ce_loss)),
                            #BCEloss=("{:5f}".format(total_bce_loss)),
                            Diceloss=("{:5f}".format(total_dice_loss)),
                            F_SOCRE=("{:5f}".format(total_f_score)),
                            lr=("{:7f}".format(get_lr(optimizer))))

    return loss, ce_loss, bce_loss, dice_loss, get_lr(optimizer)


# 测试方法
def test(model, val_loader):

    total_ce_loss   = 0
    total_bce_loss  = 0
    total_dice_loss = 0
    total_f_score   = 0
    total_loss      = 0

    model_eval = model.eval()

    tqdmbar = tqdm(val_loader)
    for batch_idx, batch in enumerate(tqdmbar):
        img, png, label = batch

        with torch.no_grad():
            img    = torch.from_numpy(img).type(torch.FloatTensor)
            png    = torch.from_numpy(png).type(torch.FloatTensor)
            label  = torch.from_numpy(label).type(torch.FloatTensor)
            # img = torch.autograd.Variable(img).type(torch.FloatTensor)
            # png = torch.autograd.Variable(png).type(torch.FloatTensor).long()
            # seg_labels = torch.autograd.Variable(seg_labels).type(torch.FloatTensor)
            # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3)

            if this_device.type == "cuda":
                img = img.to(this_device)
                png = png.to(this_device)
                label = label.to(this_device)

        # 输入测试图像
        output    = model_eval(img)

        # ce_loss   = CELOSS(output, png)
        ce_loss   = CELoss2d()(output, png)
        bce_loss  = BCELoss2d()(output, png)
        dice_loss = DiceLoss()(output, label)

        with torch.no_grad():
            _f_score = f_score(output, label)

        loss = ce_loss + bce_loss + dice_loss

        total_ce_loss   += ce_loss.item()
        total_bce_loss  += bce_loss.item()
        total_dice_loss += dice_loss.item()
        total_f_score   += _f_score.item()
        total_loss      += loss.item()

        total_ce_loss   /= (batch_idx + 1)
        total_bce_loss  /= (batch_idx + 1)
        total_dice_loss /= (batch_idx + 1)
        total_f_score   /= (batch_idx + 1)
        total_loss      /= (batch_idx + 1)

        #设置进度条左边显示的信息
        tqdmbar.set_description("Vaild_Epoch_size")
        #设置进度条右边显示的信息
        tqdmbar.set_postfix(Loss=("{:5f}".format(total_loss)),
                            CEloss=("{:5f}".format(total_ce_loss)),
                            BCEloss=("{:5f}".format(total_bce_loss)),
                            F_SOCRE=("{:5f}".format(total_f_score)),
                            Diceloss=("{:5f}".format(total_dice_loss)))

    return loss, ce_loss, bce_loss, dice_loss


# 定义命令行参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default=r'dataset/')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=2)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=2)
    parser.add_argument('--load_tread', type=int,
                        help='load data thread', default=1)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--IMGSIZE', type=list, 
                        help='IMGSIZE', default=[384, 384, 384])
    parser.add_argument('--lr', type=list, 
                        help='Learning rate', default=[0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=15)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--log_path', type=str,
                        help='log save path', default=r'log/')
    parser.add_argument('--save_mode', type=bool,
                        help='true save mode false save dic', default=True)
    parser.add_argument('--resume', type=bool,
                        help='user resume', default=True)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    parser.add_argument('--UseGPU', type=bool,
                        help='is use cuda as env', default=True)
                        
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args        = get_args()
    start_epoch = 0                   # 起始的批次
    LoadThread  = args.load_tread     # 加载数据线程数
    SaveMode    = args.save_mode      # 保存模型加参数还是只保存参数
    Resume      = args.resume         # 是否使用断点续训
    SEED        = args.seed           # 设置随机种子
    CLASSNUM    = args.nclass
    IMGSIZE     = args.IMGSIZE
    this_device = torch.device("cuda:0" if torch.cuda.is_available() and args.UseGPU else "cpu")
    
    # 为CPU设定随机种子使结果可靠，就是每个人的随机结果都尽可能保持一致
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 不同损失函数之间的调整系数，默认是均衡的
    cls_weights = np.ones([CLASSNUM], np.float32)

    # 加载日志对象
    writer   = WriteLog(writerpath=r'./log/log/')
    tfwriter = SummaryWriter(logdir=r"./log/tfboard/", comment="unet")

    # 打印列表参数
    # print(vars(args))
    writer.write(vars(args))

    loader = MakeLoader(IMGSIZE, CLASSNUM, args.batch_size)
    gen, gen_val = loader.makedataVoc()
    writer.write("数据集加载完毕")

    model = GetModel(args, writer).Createmodel(is_train=True)
    writer.write("模型创建及初始化完毕")

    if this_device.type == "cuda":
        # 为GPU设定随机种子，以便确信结果是可靠的
        # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
        torch.cuda.manual_seed(SEED)
        model = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
        #torch.cuda().manual_seed_all(SEED)
        # # 那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        model = model.to(this_device)
    
    # tfwriter.add_graph(model=model, input_to_model=IMGSIZE)
    writer.write("模型初始化完毕")

    # 测试网络结构
    # summary(model, input_size=(3, 384, 384))

    # 创建优化器
    optimizer, scheduler = CreateOptim(model, lr=args.lr[0])

    # 初始化 early_stopping 对象
    patience = args.early_stop # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, path="./savepoint/early_stopp/checkpoint.pth", 
                                    verbose=True, savemode=SaveMode)
    writer.write("优化器及早停模块加载完毕")

    if Resume:
        path = "./savepoint/model_data/UNet_2Class_NewLoss_1.pth"
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            start_epoch = checkpoint['epoch']
            if SaveMode:
                model = checkpoint['model']
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer = checkpoint['optimizer']
            writer.write("加载数据检查点，从(epoch {})开始".format(checkpoint['epoch']))
        else:
            writer.write("没有找到检查点，从(epoch 1)开始")

    # 开始训练
    tqbar = tqdm(range(start_epoch + 1, args.nepoch + 1))
    writer.write("开始训练")
    for epoch in tqbar:
        #loss, loss_cls, loss_lmmd = train_epoch(epoch, model, [tra_source_dataloader,tra_target_dataloader] , optimizer, scheduler)
        #t_correct = test(model, test_dataloader)
        
        # 训练
        loss, ce_loss, bce_loss, dice_loss, getLr = \
            fit_one_epoch(model, epoch, gen, optimizer, scheduler)
        
        # 进行测试
        test(model, gen_val)
        
        # 判断是否满足早停
        early_stopping(loss, model.train)

        # 学习率逐步变小
        scheduler.step()
        
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
        }
        saveparafilepath = "./savepoint/model_data/UNEt_2Class_NewLoss_checkpoint.pth"
        torch.save(checkpoint, saveparafilepath)
        writer.write("保存检查点完成，当前批次{}, 当然权重文件保存地址{}".format(epoch, saveparafilepath))

        # 若满足 early stopping 要求 且 当前批次>=10
        if early_stopping.early_stop and \
            epoch >= 20:
            # print("命中早停模式，当前批次{}".format(epoch))
            writer.write("命中早停模式，当前批次{}".format(epoch))
            # os.system('/root/shutdown.sh') 
            break

        # 设置进度条左边显示的信息
        tqbar.set_description("Train Epoch Count")
        # 设置进度条右边显示的信息
        tqbar.set_postfix()

    checkpoint = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
    }
    saveparafilepath = "./savepoint/model_data/checkpoint.pth"
    torch.save(checkpoint, saveparafilepath)
    writer.write("保存检查点完成，当前批次{}, 当然权重文件保存地址{}".format(epoch, saveparafilepath))

    os.system('shutdown /s /t 0')       # 0秒之后Windows关机
    # os.system('/root/shutdown.sh')    # 极客云停机代码
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name RTX2080Ti")    # 矩池云停机代码(包含保存相应环境)
    # 若释放前不需要保存环境 
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")

