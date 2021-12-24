# -*- coding: UTF-8 -*- 
import argparse
import os
import sys
import platform
import time
import pynvml

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm, trange
from loguru import logger

from train_funtion import fit_one_epoch, test_epoch

from modules.nets.getmodel import GetModel
from modules.utils.getearlystop import GetEarlyStopping
from modules.utils.getloader import GetLoader
from modules.utils.getlog import GetWriteLog
from modules.utils.getoptim import GetOptim

def set_lr(optimizer, value:float):
    """optimizer.param_groups:是长度为2的list,0表示params/lr/eps等参数，1表示优化器状态"""
    optimizer.param_groups[0]['lr'] = value

def formt_time(timecount):
    """将浮点秒转化为字符串时间"""
    _time = int(timecount)
    hour   = -1
    minute = _time // 60
    second = _time % 60
    if minute >= 60:
        hour   = minute // 60
        minute = minute % 60
    if hour == -1:
        if minute == 0:
            time_str = "{}秒".format(second)
        else:
            time_str = "{}分{}秒".format(minute, second)
    else:
        time_str = "{}小时{}分{}秒".format(hour, minute, second)

    return time_str

def makedir(path):
    """创建文件夹"""
    workpath = os.getcwd()
    if not os.path.isabs(path):
        path = os.path.join(workpath, path)

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def getgpudriver():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()           # 几块显卡
    numbers_list = []
    gpu_ids      = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)    # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        drive_name = str(pynvml.nvmlDeviceGetName(handle))[2:-1]
        gpu_model_number = str(drive_name.rpartition(" ")[-1]) \
            if str(drive_name.rpartition(" ")[-1]) != "Ti" else str(drive_name.rpartition(" ")[-2])
        numbers_list.append(gpu_model_number)
        gpu_ids.append(i)
        logger.info("当前显卡为:{}.".format(drive_name) +
                    "总显存大小{:.0f} G,已用{:.0f} G,剩余{:.0f} G".format((meminfo.total / 1024**2),
                    (meminfo.used / 1024**2), (meminfo.free / 1024**2))) #第二块显卡总的显存大小
        break # 只考虑存在所有宿主机都存在的是同一种显卡的情况

    return numbers_list, gpu_ids

# 定义命令行参数
def get_args():
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
                        help='IMGSIZE', default=[384, 384, 3])
    parser.add_argument('--lr', type=list, 
                        help='Learning rate', default=[0.001, 0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=5)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=42)    # 迷信数字
    parser.add_argument('--save_mode', type=bool,
                        help='true save mode false save dic', default=True)
    parser.add_argument('--resume', type=bool,
                        help='user resume weight', default=True)
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
    parser.add_argument("--systemtype_mac", type=bool,
                        help='net run on system, True is mac', default=False)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')  # DDP多卡参数
    args = parser.parse_args()

    return args

def main():
    args        = get_args()
    start_epoch = 0                   # 起始的批次

    # 使用window平台还是Linux平台
    args.systemtype  = True if platform.system().lower() == 'windows' else False
    args.systemtype_mac = True if platform.mac_ver()[0] != "" else False
    # 当前是否使用cuda来进行加速
    args.UseGPU = True
    this_device = torch.device("cuda:0" if torch.cuda.is_available() and args.UseGPU else "cpu")

    # 根据平台的不同，设置不同batch的大小
    if args.systemtype == True:
        args.batch_size = 1
        args.load_tread = 1
        args.UseTfBoard = False
        # args.amp        = False
    else:
        args.batch_size = 6
        args.load_tread = 16
        args.UseTfBoard = True
        # args.amp        = False

    # 不期望使用amp混合精度的列表
    hope_gpu_name     = ["960", "2080", "T4", ]
    # 获取当前GPU名称
    gpu_name, gpu_ids = getgpudriver()
    for i in range(len(gpu_name)):
        name = gpu_name[i]
        # 如果不在期望列表，使用amp混合训练
        if name not in hope_gpu_name:
            args.amp = True
    # 当设备中存在的GPU数量大于1时，开启GPU并行计算
    if torch.cuda.device_count() > 1:
        args.UseMultiGPU = True

    # 为CPU设定随机种子使结果可靠，就是每个人的随机结果都尽可能保持一致
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 不同分类之间的权重系数，默认都为1（均衡的）
    cls_weights = np.ones([args.n_class], np.float32)

    # 加载日志对象
    #logger = GetWriteLog(writerpath=MakeDir("log/log/"))  # 需注释掉最前方引用的logger库
    logger.add(makedir("log/log/") + "logfile_{time:MM-DD_HH:mm}.log", format="{time:DD Day HH:mm:ss} | {level} | {message}", filter="",
               enqueue=True, encoding='utf-8', rotation="50 MB")
    tfwriter = SummaryWriter(logdir=makedir("log/tfboard/"), comment="unet") \
        if args.systemtype == False or (args.systemtype == True and args.UseTfBoard) else None

    # 打印列表参数
    # print(vars(args))
    logger.info(vars(args))

    loader = GetLoader(args)
    gen, gen_val = loader.makedata()
    gen_target   = [1,] # loader.makedataTarget()
    logger.success("数据集加载完毕")

    modeler = GetModel(args)
    model = modeler.Createmodel(is_train=True)
    modeler.init_weights(model, "kaiming")
    logger.success("模型创建及初始化完毕")

    if this_device.type == "cuda":
        # 为GPU设定随机种子，以便确信结果是可靠的
        # os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
        torch.cuda.manual_seed(args.seed)
        # 那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        torch.backends.cudnn.benchmark = True
        # 将模型设置为GPU
        model = model.to(this_device)
        if args.UseMultiGPU:
            # 开启GPU并行化处理
            torch.backends.cudnn.deterministic = True
            model = torch.nn.DataParallel(model, gpu_ids)

    # tfwriter.add_graph(model=model, input_to_model=args.IMGSIZE)
    logger.success("模型初始化完毕")

    # 测试网络结构
    # summary(model, input_size=(args.IMGSIZE[2], args.IMGSIZE[0], args.IMGSIZE[1]), device=this_device.type)

    # 创建优化器
    optimizer, scheduler = GetOptim(model, lr=args.lr[0])

    # 初始化 early_stopping 对象
    patience = args.early_stop # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    path = makedir("savepoint/early_stopp/")
    early_stopping = GetEarlyStopping(patience, path=path + "checkpoint.pth",
                                      verbose=True, savemode=args.save_mode)
    logger.success("优化器及早停模块加载完毕")

    if args.resume:
        path = "./savepoint/model_data/SmarUNEt_DiceCELoss_KMInit____.pth"
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            start_epoch = checkpoint['epoch'] if checkpoint['epoch'] != -1 else 0
            if args.save_mode:
                model = checkpoint['model']
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer = checkpoint['optimizer']
            set_lr(optimizer, 0.001)
            logger.info("加载数据检查点，从(epoch {})开始".format(checkpoint['epoch']))
        else:
            logger.info("没有找到检查点，从(epoch 1)开始")

    logger.info("注意: 没有使用tfboard记录数据") if tfwriter is None else logger.success("注意: 使用tfboard记录数据")
    logger.success("注意: 使用了amp混合精度训练") if args.amp else logger.info("注意: 没有使用amp混合精度训练")
    logger.success("注意: 使用了GPU加速训练") if this_device.type == "cuda" else logger.info("注意: 没有使用GPU加速训练")
    logger.info("注意: 系统检测多GPU，并行训练") if args.UseMultiGPU == True else logger.success("注意: 单卡训练")

    # 将最优损失设置为无穷大
    best_loss = float("inf")
    # 将优化器针对的损失进行特殊距离
    best_opt_loss = -float("inf")

    # 定义一些非关键参数，以字典的形式传入训练中
    # 用于输出相关的日志信息
    para_kargs = {
        "tf_writer" : tfwriter,
        "device" : this_device,
        "gpuids" : gpu_ids,
        "log" : logger,
        "CLASSNUM" : args.n_class,
        "IMGSIZE" : args.IMGSIZE,
        "optimizer" : optimizer,
        "amp" : args.amp,
        "cls_weight" : cls_weights,
        "this_epoch" : int,
    }

    # 开始训练
    tqbar = tqdm(range(start_epoch + 1, args.nepoch + 1))
    logger.success("开始训练")
    for epoch in tqbar:
        #loss, loss_cls, loss_lmmd = train_epoch(epoch, model, [tra_source_dataloader,tra_target_dataloader] , optimizer, scheduler)
        #t_correct = test(model, test_dataloader)
        para_kargs["this_epoch"] = epoch
        # 训练
        # 返回值按照 0/总loss, 1/count, 2/celoss, 3/bceloss, 4/diceloss, 5/floss, 6/lr
        # time_start = time.time()
        # ret_train = \
        #     fit_one_epoch(model, (gen, gen_target), **para_kargs)
        # time_end = time.time()
        #
        # # 每轮训练输出一些日志信息
        # logger.info("第{}轮训练完成,本轮训练轮次{},耗时{},最终损失为{}".format(epoch, ret_train[1],
        #                                                     formt_time((time_end - time_start)),
        #                                                     ret_train[0].item()))

        # 进行测试
        ret_val = \
            test_epoch(model, (gen_val, gen_target), **para_kargs)

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
        path = makedir("savepoint/model_data/")
        saveparafilepath = path + "SmarUNEt_NewGN_NewGAM"
        # 判断当前损失是否变小，变小才进行保存参数
        # 注意ret[0]是tensor格式，ret[1]才是平均损失（损失累加除以轮次）
        # 使用的是验证集上的损失，如果验证集损失一直在下降也是，说明模型还在训练
        if ret_val[1] < best_loss:
            best_loss = ret_val[1]
            torch.save(checkpoint, saveparafilepath)
            logger.info("保存检查点完成, 当前批次{}, 保存最优参数权重文件{}".format(epoch, saveparafilepath + "_bestepoch" + ".pth"))
        # 如果不是最优的，直接保存默认的
        torch.save(checkpoint, saveparafilepath + ".pth")
        logger.success("完成当前批次{}训练, 损失值较上一轮没有减小，正常保存模型".format(epoch))

        # 若满足 early stopping 要求 且 当前批次>=10
        if early_stopping.get_early_stop_state:
            logger.info("命中早停模式，当前批次{}".format(epoch))
            if epoch >= 5:
                logger.info("停止训练，当前批次{}".format(epoch))
                # os.system('/root/shutdown.sh')
                break

        # 设置进度条左边显示的信息
        tqbar.set_description("Train Epoch Count")
        # 设置进度条右边显示的信息
        tqbar.set_postfix()

    # 任务已经结束了，保存一个最终版本的参数
    checkpoint = {
        'epoch': para_kargs["this_epoch"],
        'model': model,
        'optimizer': optimizer,
    }
    path = makedir("savepoint/model_data/")
    saveparafilepath = path + "checkpoint.pth"
    torch.save(checkpoint, saveparafilepath)
    logger.success("保存检查点完成，当前批次{}, 当然权重文件保存地址{}".format(para_kargs["this_epoch"], saveparafilepath))

    os.system('shutdown /s /t 0')       # 0秒之后Windows关机
    # os.system('/root/shutdown.sh')    # 极客云停机代码
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node -save -name RTX2080Ti")    # 矩池云停机代码(包含保存相应环境)
    # 若释放前不需要保存环境
    # os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")

if __name__ == '__main__':
    main()

