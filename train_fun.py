# -*- coding: utf-8 -*-
# @Time     : 2021/12/20 下午 09:26
# @Author   : zhonzxad
# @File     : train_fun.py
import argparse
import os
import sys
import platform
import time
import pynvml
from loguru import logger

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def get_lr(optimizer):
    """获取学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, value:float):
    """optimizer.param_groups：是长度为2的list,0表示params/lr/eps等参数，1表示优化器状态"""
    optimizer.param_groups[0]['lr'] = value

def set_tqdm_post(vals, batch_indx, optimizer):
    """按照固定的顺序对loss相关信息进行输出"""
    names = ["Loss", "CEloss", "BCEloss", "Diceloss", "F_SOCRE", ]
    info = ""
    for i in range(len(vals)):
        if vals[i] > 0:
            info += ("{}={:.5f},".format(names[i], (vals[i] / batch_indx)))
        elif vals[i] < 0:
            logger.warning("序列化损失函数时发生错误,存在{}损失值小于0的情况".format(names[i]))
        else:
            pass

    info += ("lr={:.7f}".format(get_lr(optimizer)))

    return info
    # tqdmbar.set_postfix(Loss=("{:5f}".    format(total_loss / (batch_idx + 1))),
    #                     CEloss=("{:5f}".  format(total_ce_loss / (batch_idx + 1))),
    #                     BCEloss=("{:5f}". format(total_bce_loss / (batch_idx + 1))),
    #                     Diceloss=("{:5f}".format(total_dice_loss / (batch_idx + 1))),
    #                     F_SOCRE=("{:5f}". format(total_f_score / (batch_idx + 1))),
    #                     lr=("{:7f}".      format(get_lr(optimizer))))

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
    deviceCount = pynvml.nvmlDeviceGetCount()           # 几块显卡
    numbers_list = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        drive_name = str(pynvml.nvmlDeviceGetName(handle))[2:-1]
        gpu_model_number = str(drive_name.rpartition(" ")[-1]) \
                        if str(drive_name.rpartition(" ")[-1]) != "Ti" else str(drive_name.rpartition(" ")[-2])
        numbers_list.append(gpu_model_number)
        logger.info("当前显卡为:{}.".format(pynvml.nvmlDeviceGetName(handle))  +
                    "总显存大小{:.0f}G,已用{:.0f}G,剩余{:.0f}G".format((meminfo.total / 1024**2), (meminfo.used / 1024**2), (meminfo.free / 1024**2))) #第二块显卡总的显存大小

    return numbers_list
