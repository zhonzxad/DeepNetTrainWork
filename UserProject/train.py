# -*- coding: UTF-8 -*- 
import argparse
import os
import platform
import sys
import time
from typing import List, Tuple

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pynvml
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from train_in_epoch import test_in_epoch, train_in_epoch
from UserProject.modules.nets.funtion.user_summary import count_param
from UserProject.modules.nets.getmodel import GetModel
from UserProject.modules.utils.getearlystop import GetEarlyStopping
from UserProject.modules.utils.getloader import GetLoader
from UserProject.modules.utils.getlog import GetWriteLog
from UserProject.modules.utils.getoptim import GetOptim


def set_lr(optimizer, value: float):
    """optimizer.param_groups:是长度为2的list,0表示params/lr/eps等参数，1表示优化器状态"""
    optimizer.param_groups[0]['lr'] = value


def format_time(timecount: float) -> str:
    """将浮点秒转化为字符串时间"""
    _time = int(timecount)
    hour = -1
    minute = _time // 60
    second = _time % 60
    if minute >= 60:
        hour = minute // 60
        minute = minute % 60
    if hour == -1:
        if minute == 0:
            time_str = "{}秒".format(second)
        else:
            time_str = "{}分{}秒".format(minute, second)
    else:
        time_str = "{}小时{}分{}秒".format(hour, minute, second)

    return time_str


def makedir(path: str = "") -> str:
    """创建文件夹"""
    hope_path = path
    # 特例判断
    # python里的str是不可变对象，因此不存在修改一个字符串这个说法，任何对字符串的运算都会产生一个新字符串作为结果
    if hope_path == "":
        return ""

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


def get_gpu_info() -> Tuple[List[str], List[int]]:
    """处理显卡相关参数的信息"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()  # 几块显卡
    numbers_list = []
    gpu_ids = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        drive_name = str(pynvml.nvmlDeviceGetName(handle))[2:-1]
        gpu_model_number = str(drive_name.rpartition(" ")[-1]) \
            if str(drive_name.rpartition(" ")[-1]) != "Ti" else str(drive_name.rpartition(" ")[-2])
        numbers_list.append(gpu_model_number)
        gpu_ids.append(i)
        logger.info("当前显卡为:{}.".format(drive_name) +
                    "总显存大小{:.0f} MB,已用{:.0f} MB,剩余{:.0f} MB".format((meminfo.total / 1024 ** 2),
                                                                 (meminfo.used / 1024 ** 2),
                                                                 (meminfo.free / 1024 ** 2)))  # 第二块显卡总的显存大小
        break  # 只考虑存在所有宿主机都存在的是同一种显卡的情况

    return numbers_list, gpu_ids


def get_args():
    """定义命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_class', type=int,
                        help='Number of classes', default=2)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=1)
    parser.add_argument('--load_tread', type=int,
                        help='load data thread', default=1)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=50)
    parser.add_argument('--IMGSIZE', type=list,
                        help='IMGSIZE', default=[768, 768, 3])
    parser.add_argument('--lr', type=list,
                        help='Learning rate', default=[0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=5)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=42)  # 迷信数字
    parser.add_argument('--save_mode', type=bool,
                        help='true save mode false save dic', default=True)
    parser.add_argument('--resume', type=bool,
                        help='user resume weight', default=False)
    parser.add_argument('--start_epoch', type=int,
                        help='the epoch start', default=0)# 起始的批次
    parser.add_argument('--UseGPU', type=bool,
                        help='is use cuda as env', default=True)
    parser.add_argument('--UseMultiGPU', type=bool,
                        help='is use Multi cuda as env', default=False)
    parser.add_argument('--UseTfBoard', type=bool,
                        help='is use record tf board', default=False)
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision', default=False)
    parser.add_argument("--systemtype", type=bool,
                        help='net run on system, True is windows', default=True)
    parser.add_argument("--is_use_sysmac", type=bool,
                        help='net run on system, True is mac', default=False)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')  # DDP多卡参数
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # 使用window平台还是Linux平台, 为True表示为Windows平台
    args.systemtype = True if platform.system().lower() == 'windows' else False
    args.is_use_sysmac = True if platform.mac_ver()[0] != "" else False

    # 根据平台的不同，设置不同batch的大小，以及是否使用GPU加速
    # True表示Windows平台
    if args.systemtype:
        args.batch_size = 1
        args.load_tread = 1
        args.UseTfBoard = False
        # args.amp        = False
        args.UseGPU = True
    # mac平台
    elif args.is_use_sysmac:
        args.batch_size = 1
        args.load_tread = 1
        args.UseTfBoard = False
        # args.amp        = False
        args.UseGPU = False
        args.mac_device = "mps"
    else:
        args.batch_size = 6
        args.load_tread = 16
        args.UseTfBoard = True
        # args.amp        = False
        args.UseGPU = True

    # 判断是否使用GPU加速
    this_device = torch.device("cuda:0" if torch.cuda.is_available() and args.UseGPU else "cpu")
    if this_device.type != "cuda" and args.is_use_sysmac == True:
        this_device = torch.device(args.mac_device)

    # 判断是否使用amp加速
    if this_device.type == "cuda" or args.amp == True:
        # 不期望使用amp混合精度的列表
        hope_gpu_not_use_amp = ["960", "2080", "T4", ]
        hope_gpu_use_amp = ["3080", "9000", "A5000", ]
        # 获取当前GPU名称
        gpu_name, gpu_ids = get_gpu_info()
        for i in range(len(gpu_name)):
            name = gpu_name[i]
            # # 如果不在期望列表，使用amp混合训练
            # if name not in hope_gpu_not_use_amp:
            #     args.amp = True
            # 如果在期望列表，使用amp混合训练
            if name in hope_gpu_use_amp:
                args.amp = True
        # 当设备中存在的GPU数量大于1时，开启GPU并行计算
        if torch.cuda.device_count() > 1:
            args.UseMultiGPU = True
    else:
        gpu_name = ""
        gpu_ids = 0

    # 为CPU设定随机种子使结果可靠，就是每个人的随机结果都尽可能保持一致
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 不同分类之间的权重系数，默认都为1（均衡的）
    cls_weights = np.ones([args.n_class], np.float32)

    # 加载日志对象
    # logger = GetWriteLog(writerpath=MakeDir("logs/log/"))  # 需注释掉最前方引用的logger库
    log_file_path = makedir("logs/log/")
    logger.add(log_file_path + "/logfile_{time:MM-DD_HH:mm}.txt",
               format="{time:DD Day HH:mm:ss} | {level} | {message}", filter="",
               enqueue=True, encoding='utf-8', rotation="50 MB")

    # 加载tensorboard记录日志对象
    tfwriter = SummaryWriter(logdir=makedir("logs/tfboard/"), comment="unet") if args.UseTfBoard else None

    # 打印列表参数
    # print(vars(args))
    logger.info(vars(args))

    # 创建数据迭代器dataloader
    loader = GetLoader(args)
    gen, gen_val = loader.makedata()
    gen_target = [1, ]  # loader.makedataTarget()
    logger.success("数据集加载完毕")

    # 创建自定义模型参数
    modeler = GetModel(args)
    model = modeler.Createmodel(is_train=True)

    # 将测试模型参数量挪到刚创建模型之后，防止后续使用CUDA报错超内存
    # 测试网络结构
    # summary(model, input_size=(args.IMGSIZE[2], args.IMGSIZE[0], args.IMGSIZE[1]), device=this_device.type)
    # count = count_param(model=model)

    # 初始化权重
    modeler.init_weights(model, "kaiming")
    # 是否使用预训练参数权重继续训练
    args.resume = False
    if args.resume:
        path = "savepoint/model_data/SmarUNEt_DiceCELoss_KMInit____.pth"
        if os.path.exists(path) and os.path.isfile(path):
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch'] if checkpoint['epoch'] != -1 else 0
            if args.save_mode:
                model = checkpoint['model']
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer = checkpoint['optimizer']
            set_lr(optimizer, 0.001)
            logger.info("加载数据检查点，从(epoch {})开始".format(checkpoint['epoch']))
        else:
            logger.info("没有找到检查点，从(epoch 1)开始")
    else:
        # G:\Py_Debug\GraduationProject\SignalModel\UNet_Pytorch\model_data\unet_resnet_voc.pth
        model_pretrain_path = r''
        if os.path.exists(model_pretrain_path):
            logger.info('Load weights {}.'.format(model_pretrain_path))
            model_dict = model.state_dict()
            pre_trained_dict = torch.load(model_pretrain_path, map_location=this_device)
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pre_trained_dict)
            model.load_state_dict(model_dict)
    logger.success("模型创建及初始化完毕")

    if this_device.type != "cpu":
        # 将模型设置为GPU
        model = model.to(this_device)
    if this_device.type == "cuda":
        # 为GPU设定随机种子，以便确信结果是可靠的
        # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
        torch.cuda.manual_seed(args.seed)
        # 那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        torch.backends.cudnn.benchmark = True
        if args.UseMultiGPU:
            # 开启GPU并行化处理
            torch.backends.cudnn.deterministic = True
            model = torch.nn.DataParallel(model, gpu_ids)

    # tfwriter.add_graph(model=model, input_to_model=args.IMGSIZE)
    logger.success("模型初始化完毕")

    # 创建优化器
    optimizer, scheduler = GetOptim(model, lr=args.lr[0])

    # 初始化 early_stopping 对象
    patience = args.early_stop  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    path = makedir("logs/savepoint/early_stopp/")
    early_stopping = GetEarlyStopping(patience, path=os.path.join(path, "checkpoint.pth"),
                                      verbose=True, savemode=args.save_mode)
    logger.success("优化器及早停模块加载完毕")

    logger.info("注意: 没有使用tfboard记录数据") if tfwriter is None else logger.success("注意: 使用tfboard记录数据")
    logger.success("注意: 使用了amp混合精度训练") if args.amp else logger.info("注意: 没有使用amp混合精度训练")
    logger.success("注意: 使用了GPU加速训练") if this_device.type != "cpu" else logger.info("注意: 没有使用GPU加速训练")
    logger.info("注意: 系统检测多GPU，并行训练") if args.UseMultiGPU == True else logger.success("注意: 单卡训练")

    # 将最优损失设置为无穷大
    best_loss = float("inf")
    # 将优化器针对的损失进行特殊距离
    best_opt_loss = -float("inf")

    # 定义一些非关键参数，以字典的形式传入训练中
    # 用于输出相关的日志信息
    para_kargs = {
        "tf_writer" : tfwriter,
        "device"    : this_device,
        "gpuids"    : gpu_ids,
        "log"       : logger,
        "CLASSNUM"  : args.n_class,
        "IMGSIZE"   : args.IMGSIZE,
        "optimizer" : optimizer,
        "amp"       : args.amp,
        "cls_weight": cls_weights,
        "this_epoch": int,          # args.start_epoch,
    }

    # 开始训练
    tqbar = tqdm(range(args.start_epoch + 1, args.nepoch + 1))
    logger.success("开始训练")
    for epoch in tqbar:
        # loss, loss_cls, loss_lmmd =
        # train_epoch(epoch, model, [tra_source_dataloader,tra_target_dataloader] , optimizer, scheduler)
        # t_correct = test(model, test_dataloader)
        para_kargs["this_epoch"] = epoch

        # 训练
        logger.info("开始进行训练")
        # 返回值按照 0/总loss(float), 1/count(int), 2/[loss], 3/lr
        # loss = tensor格式 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss
        time_start = time.time()
        ret_train = \
            train_in_epoch(model, (gen, gen_target), **para_kargs)
        time_end = time.time()

        # 每轮训练输出一些日志信息
        train_time = (time_end - time_start)
        logger.info("第{}轮训练完成,本轮训练轮次{},耗时{}\n".format(epoch, ret_train[1], train_time))

        # 进行测试
        logger.info("开始进行验证")
        # 返回值按照 0/总loss(float), 1/count(int), 2/[loss，tensor]
        # loss = tensor格式 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss
        time_start = time.time()
        ret_val = \
            test_in_epoch(model, (gen_val, gen_target), **para_kargs)
        time_end = time.time()

        val_time = (time_end - time_start)
        logger.info(
            "本轮训练总耗时{}, 最终测试集损失为{},验证集损失{}".format(format_time(train_time + val_time), ret_train[0], ret_val[0]))

        # 判断是否满足早停
        early_stopping(ret_val[0], model.eval)

        # 只有当前学习率较之前不够优异时，才去优化学习率，进行更细粒度的调节
        # if ret_val[0] >= best_opt_loss:
        #     best_opt_loss = ret_val[0]
        scheduler.step(ret_train[2][0])

        # 一些保存的参数
        checkpoint = {
            'epoch'    : epoch,
            'model'    : model,
            'optimizer': optimizer,
            'loss'     : ret_val,
        }
        saveparafilepath = makedir("logs/savepoint/model_data/")
        file_name = "SmarUNet_20220223.pth"
        # 判断当前损失是否变小，变小才进行保存参数
        # 注意ret[0]是tensor格式，ret[1]才是平均损失（损失累加除以轮次）
        # 使用的是验证集上的损失，如果验证集损失一直在下降也是，说明模型还在训练
        if ret_val[0] < best_loss:
            best_loss = ret_val[0]
            beat_file_path = os.path.join(saveparafilepath, file_name.replace(".pth", "_best.pth"))
            torch.save(checkpoint, beat_file_path)
            logger.info("保存检查点完成, 当前批次{}, 保存最优参数权重文件{}".format(epoch, beat_file_path))
        else:
            # 如果不是最优的，直接保存默认的
            total_file_path = os.path.join(saveparafilepath, file_name)
            torch.save(checkpoint, total_file_path)
            logger.success("完成当前批次{}训练, 损失值较上一轮没有减小，正常保存模型文件".format(epoch, total_file_path))

        # 若满足 early stopping 要求 且 当前批次>=10
        if early_stopping.get_early_stop_state:
            logger.info("命中早停模式，当前批次{}".format(epoch))
            if epoch >= 15:
                logger.info("停止训练，当前批次{}".format(epoch))
                # os.system('/root/shutdown.sh')
                break

        # 设置进度条左边显示的信息
        tqbar.set_description("Train Epoch Count")
        # 设置进度条右边显示的信息
        tqbar.set_postfix()

    # 任务已经结束了，保存一个最终版本的参数
    checkpoint = {
        'epoch'    : para_kargs["this_epoch"],
        'model'    : model,
        'optimizer': optimizer,
    }
    path = makedir("logs/savepoint/model_data/")
    saveparafilepath = os.path.join(path, "checkpoint.pth")
    torch.save(checkpoint, saveparafilepath)
    logger.success("保存检查点完成，当前批次{}, 当然权重文件保存地址{}".format(para_kargs["this_epoch"], saveparafilepath))

    os.system('shutdown /s /t 0')  # 0秒之后Windows关机
    # os.system('/root/shutdown.sh')    # 极客云停机代码
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name RTX2080Ti")    # 矩池云停机代码(包含保存相应环境)
    # 若释放前不需要保存环境
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")


if __name__ == '__main__':
    main()
