# -*- coding: utf-8 -*-
# @Time     : 2021/12/24 下午 01:57
# @Author   : zhonzxad
# @File     : DataParallel.py
"""
DataParallel（DP）：Parameter Server模式，一张卡为reducer，实现也超级简单，一行代码。
DistributedDataParallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。
第一种方法容易出现负载不均衡的情况
第二种适用于多级多卡，修改后也可适用于单机多卡
http://www.manongjc.com/detail/16-egzyzijukzuqmwv.html
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torch.distributed
# import torch.distributed as dist


class DDP():
    """
    group：即进程组。默认情况下，只有一个组，一个 job 即为一个组，也即一个 world。
        当需要进行更加精细的通信时，可以通过 new_group 接口，使用 word 的子集，创建新组，用于集体通信等。
    world size ：表示全局进程个数。
    rank：表示进程序号，用于进程间通讯，表征进程优先级。rank = 0 的主机为 master 节点。
    local_rank：进程内，GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。比方说， rank = 3，local_rank = 0 表示第 3 个进程内的第 1 块 GPU。

    启动范例
    pytorch 为我们提供了 torch.distributed.launch 启动器，用于在命令行分布式地执行 python 文件。
    --nproc_per_node参数指定为当前主机创建的进程数。一般设定为=NUM_GPUS_YOU_HAVE当前主机的 GPU 数量，每个进程独立执行训练脚本。
    这里是单机多卡，所以node=1，就是一台主机，一台主机上--nproc_per_node个进程
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
    """
    
    def __init__(self, local_rank, device_ids, batch_size):
        super(DDP, self).__init__()
        # 1) 初始化
        torch.distributed.init_process_group(backend="nccl")

        # 2） 配置每个进程的gpu
        self.local_rank = local_rank
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        ngpus_per_node = len(device_ids)
        self.batch_size = int(batch_size / ngpus_per_node)

        # ps 检查nccl是否可用
        # torch.distributed.is_nccl_available ()

    def get_device(self):
        return self.device

    def get_local_rank(self):
        return self.local_rank

    def get_batchsize(self):
        return self.batch_size