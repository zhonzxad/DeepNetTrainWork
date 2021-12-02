# -*- coding: UTF-8 -*- 
import argparse
import os
import sys
# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
os.chdir(sys.path[0])
sys.path.append("..")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from models.Unit.getloader import GetLoader
from models.Net.getmodel import GetModel
from models.Unit.getloss import loss_func
from models.Unit.getoptim import GetOptim
from models.Unit.Getlog import GetWriteLog
from models.Unit.getearlystop import GetEarlyStopping


def get_lr(optimizer):
    """获取学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, value:float):
    optimizer.param_groups[0]['lr'] = value

def MakeDir(path):
    """创建文件夹"""
    workpath = os.getcwd()
    if not os.path.isabs(path):
        path = os.path.join(workpath, path)

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def fit_one_epoch(net, dataloader, **kargs):
    """"定义训练每一个epoch的步骤"""
    total_ce_loss   = 0
    total_bce_loss  = 0
    total_dice_loss = 0
    total_f_score   = 0
    total_loss      = 0

    amp         = kargs["amp"]
    this_device = kargs["device"]
    optimizer   = kargs["optimizer"]
    tfwriter    = kargs["tf_writer"]
    cls_weights = kargs["cls_weight"]

    # 定义网络为训练模式
    model_train = net.train()

    # 创建混合精度训练
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    tqdmbar = tqdm(dataloader, total=len(dataloader), leave=False, mininterval=0.2)
    for batch_idx, batch in enumerate(tqdmbar):

        img, png, label = batch
        # print("\nNo in RangeNet img shape is {} || png shape is {}".format(img.shape, png.shape))

        with torch.no_grad():
            # img = torch.autograd.Va   riable(torch.from_numpy(img).type(torch.FloatTensor))
            # png = torch.autograd.Variable(torch.from_numpy(png).type(torch.FloatTensor)).long()
            # seg_labels = torch.autograd.Variable(torch.from_numpy(seg_labels).type(torch.FloatTensor))
            img     = torch.from_numpy(img).type(torch.FloatTensor)
            png     = torch.from_numpy(png).type(torch.FloatTensor)
            label   = torch.from_numpy(label).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            # img = torch.autograd.Variable(img).type(torch.FloatTensor)
            # png = torch.autograd.Variable(png).type(torch.FloatTensor)
            # seg_labels = torch.autograd.Variable(seg_labels).type(torch.FloatTensor)
            # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3)
            # logger.write("\n img shape is {} || png shape is {} || seg_labels shape is {}".format(img.shape, png.shape, seg_labels.shape))

            if this_device.type == "cuda":
                img     = img.to(this_device)
                png     = png.to(this_device)
                label   = label.to(this_device)
                weights = weights.to(this_device)

        # 所有梯度为0
        optimizer.zero_grad()

        # 混合精度计算
        with torch.cuda.amp.autocast(enabled=amp):
            # 网络计算
            output = model_train(img)
            # 计算损失
            # print("\n output shape is {} || png shape is {}".format(output.shape, png.shape))
            # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
            loss = loss_func(output, png, label, weights, this_device)

            # 误差反向传播
            grad_scaler.scale(loss[0]).backward()
            # 优化梯度
            grad_scaler.step(optimizer)
            grad_scaler.update()

        total_loss      += loss[0].item()
        total_ce_loss   += loss[1].item()
        total_bce_loss  += loss[2].item()
        total_dice_loss += loss[3].item()
        total_f_score   += loss[4].item()

        # 写tensorboard
        tags = ["train_loss", "CEloss", "BCEloss", "Diceloss", "f_score", "lr", "accuracy"]
        if tfwriter != None:
            tfwriter.add_scalar(tags[0],     total_loss / (batch_idx + 1))#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[1],     total_ce_loss / (batch_idx + 1))#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[2],     total_bce_loss / (batch_idx + 1))#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[3],     total_dice_loss / (batch_idx + 1))#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[4],     total_f_score / (batch_idx + 1))#, epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[5], get_lr(optimizer))#, epoch*(batch_idx + 1))

        #设置进度条左边显示的信息
        tqdmbar.set_description("Train {}".format(epoch))
        #设置进度条右边显示的信息
        tqdmbar.set_postfix(Loss=("{:5f}".    format(total_loss / (batch_idx + 1))),
                            CEloss=("{:5f}".  format(total_ce_loss / (batch_idx + 1))),
                            BCEloss=("{:5f}". format(total_bce_loss / (batch_idx + 1))),
                            Diceloss=("{:5f}".format(total_dice_loss / (batch_idx + 1))),
                            F_SOCRE=("{:5f}". format(total_f_score / (batch_idx + 1))),
                            lr=("{:7f}".      format(get_lr(optimizer))))

    # 返回值按照 0/总loss, 1/count, 2/celoss, 3/bceloss, 4/diceloss, 5/floss, 6/lr
    return [loss[0], (batch_idx + 1), loss[1], loss[2], loss[3], loss[4], get_lr(optimizer)]


def test(net, val_loader, **kargs):
    """测试方法"""
    total_ce_loss   = 0
    total_bce_loss  = 0
    total_dice_loss = 0
    total_f_score   = 0
    total_loss      = 0

    amp         = kargs["amp"]
    this_device = kargs["device"]
    optimizer   = kargs["optimizer"]
    tfwriter    = kargs["tf_writer"]
    cls_weights = kargs["cls_weight"]

    model_eval = net.eval()

    tqdmbar = tqdm(val_loader)
    for batch_idx, batch in enumerate(tqdmbar):
        img, png, label = batch

        with torch.no_grad():
            img     = torch.from_numpy(img).type(torch.FloatTensor)
            png     = torch.from_numpy(png).type(torch.FloatTensor)
            label   = torch.from_numpy(label).type(torch.FloatTensor)
            weights = torch.from_numpy(cls_weights)
            # img = torch.autograd.Variable(img).type(torch.FloatTensor)
            # png = torch.autograd.Variable(png).type(torch.FloatTensor).long()
            # seg_labels = torch.autograd.Variable(seg_labels).type(torch.FloatTensor)
            # seg_labels = seg_labels.transpose(1, 3).transpose(2, 3)

            if this_device.type == "cuda":
                img     = img.to(this_device)
                png     = png.to(this_device)
                label   = label.to(this_device)
                weights = weights.to(this_device)

        # 输入测试图像
        output    = model_eval(img)

        # 计算损失
        # print("\n output shape is {} || png shape is {}".format(output.shape, png.shape))
        # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
        loss = loss_func(output, png, label, weights, this_device)

        total_loss      += loss[0].item()
        total_ce_loss   += loss[1].item()
        total_bce_loss  += loss[2].item()
        total_dice_loss += loss[3].item()
        total_f_score   += loss[4].item()

        #设置进度条左边显示的信息
        tqdmbar.set_description("Vaild_Epoch_size")
        #设置进度条右边显示的信息
        tqdmbar.set_postfix(Loss=("{:5f}".format(total_loss / (batch_idx + 1))),
                            CEloss=("{:5f}".format(total_ce_loss / (batch_idx + 1))),
                            BCEloss=("{:5f}".format(total_bce_loss / (batch_idx + 1))),
                            F_SOCRE=("{:5f}".format(total_f_score / (batch_idx + 1))),
                            Diceloss=("{:5f}".format(total_dice_loss / (batch_idx + 1))))

    # 返回值按照 0/总loss, 1/count, 2/celoss, 3/bceloss, 4/diceloss, 5/floss
    return [loss[0], (batch_idx + 1), loss[1], loss[2], loss[3], loss[4]]


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
                        help='IMGSIZE', default=[384, 384, 3])
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
                        help='user resume weight', default=True)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    parser.add_argument('--UseGPU', type=bool,
                        help='is use cuda as env', default=True)
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision', default=False)

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
    logger   = GetWriteLog(writerpath=MakeDir("log/log/"))
    tfwriter = SummaryWriter(logdir=MakeDir("log/tfboard/"), comment="unet")

    # 打印列表参数
    # print(vars(args))
    logger.write(vars(args))

    loader = GetLoader(IMGSIZE, CLASSNUM, args.batch_size, args.load_tread)
    gen, gen_val = loader.makedata()
    logger.write("数据集加载完毕")

    modelClass = GetModel((IMGSIZE, CLASSNUM), logger)
    model = modelClass.Createmodel(is_train=True)
    logger.write("模型创建及初始化完毕")

    if this_device.type == "cuda":
        # 为GPU设定随机种子，以便确信结果是可靠的
        # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
        torch.cuda.manual_seed(SEED)
        cudnn.benchmark = True
        # 那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 将模型设置为GPU
        model = model.to(this_device)
        model = torch.nn.DataParallel(model)
    
    # tfwriter.add_graph(model=model, input_to_model=IMGSIZE)
    logger.write("模型初始化完毕")

    # 测试网络结构
    summary(model, input_size=(IMGSIZE[2], IMGSIZE[0], IMGSIZE[1]))

    # 创建优化器
    optimizer, scheduler = GetOptim(model, lr=args.lr[0])

    # 初始化 early_stopping 对象
    patience = args.early_stop # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    path = MakeDir("savepoint/early_stopp/")
    early_stopping = GetEarlyStopping(patience, path=path + "checkpoint.pth",
                                      verbose=True, savemode=SaveMode)
    logger.write("优化器及早停模块加载完毕")

    if Resume:
        path = "./savepoint/model_data/UNEt_DiceCELoss_KMInit___.pth"
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            start_epoch = checkpoint['epoch']
            if SaveMode:
                model = checkpoint['model']
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer = checkpoint['optimizer']
            set_lr(optimizer, 0.001)
            logger.write("加载数据检查点，从(epoch {})开始".format(checkpoint['epoch']))
        else:
            logger.write("没有找到检查点，从(epoch 1)开始")

    # 将最优损失设置为无穷大
    best_loss = float("inf")
    # 将优化器针对的损失进行特殊距离
    best_opt_loss = -float("inf")

    # 定义一些非关键参数，用于输出相关的日志信息
    para_kargs = {
        "tf_writer" : tfwriter,
        "device" : this_device,
        "log" : logger,
        "CLASSNUM" : args.nclass,
        "IMGSIZE" : args.IMGSIZE,
        "optimizer" : optimizer,
        "amp" : args.amp,
        "cls_weight" : cls_weights,
        "this_epoch" : int,
    }

    # 开始训练
    tqbar = tqdm(range(start_epoch + 1, args.nepoch + 1))
    logger.write("开始训练")
    for epoch in tqbar:
        #loss, loss_cls, loss_lmmd = train_epoch(epoch, model, [tra_source_dataloader,tra_target_dataloader] , optimizer, scheduler)
        #t_correct = test(model, test_dataloader)
        para_kargs["this_epoch"] = epoch
        # 训练
        # 返回值按照 0/总loss, 1/count, 2/celoss, 3/bceloss, 4/diceloss, 5/floss, 6/lr
        ret_train = \
            fit_one_epoch(model, gen, **para_kargs)
        
        # 进行测试
        ret_val = \
            test(model, gen_val, **para_kargs)
        
        # 判断是否满足早停
        early_stopping(ret_val[0], model.eval)

        # 只有当前学习率较之前不够优异时，才去优化学习率，进行更细粒度的调节
        # 依据测试数据级中dice损失来进行优化
        if ret_val[5] >= best_opt_loss:
            best_opt_loss = ret_val[5]
            scheduler.step(ret_val[5])

        # 一些保存的参数
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'loss' : ret_val,
        }
        path = MakeDir("savepoint/model_data/")
        saveparafilepath = path + "SmarUNEt_DiceCELoss_KMInit.pth"
        # 判断当前损失是否变小，变小才进行保存参数
        # 注意ret[0]是tensor格式，ret[1]才是平均损失（损失累加除以轮次）
        # 使用的是验证集上的损失，如果验证集损失一直在下降也是，说明模型还在训练
        if ret_val[1] < best_loss:
            best_loss = ret_val[1]
            torch.save(checkpoint, saveparafilepath)
            logger.write("保存检查点完成，当前批次{}, 权重文件保存地址{}".format(epoch, saveparafilepath))
        else:
            logger.write("完成当前批次{}训练, 损失值较上一轮没有减小，未保存模型".format(epoch))

        # 若满足 early stopping 要求 且 当前批次>=10
        if early_stopping.early_stop:
            logger.write("命中早停模式，当前批次{}".format(epoch))
            if epoch >= 5:
                logger.write("停止训练，当前批次{}".format(epoch))
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
        'loss' : ret_val,
    }
    path = MakeDir("savepoint/model_data/")
    saveparafilepath = path + "checkpoint.pth"
    torch.save(checkpoint, saveparafilepath)
    logger.write("保存检查点完成，当前批次{}, 当然权重文件保存地址{}".format(epoch, saveparafilepath))

    os.system('shutdown /s /t 0')       # 0秒之后Windows关机
    # os.system('/root/shutdown.sh')    # 极客云停机代码
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name RTX2080Ti")    # 矩池云停机代码(包含保存相应环境)
    # 若释放前不需要保存环境 
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")

