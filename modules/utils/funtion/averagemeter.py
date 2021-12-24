# -*- coding: utf-8 -*-
# @Time     : 2021/12/24 上午 10:33
# @Author   : zhonzxad
# @File     : averagemeter.py

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.count = 0
        self.sum   = 0
        self.this_val   = 0
        self.avg   = 0

        self.reset()

    def reset(self):
        self.this_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
<<<<<<< HEAD:modules/utils/module/funtion.py
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
=======
        self.this_val = val
        self.sum     += val
        self.count   += n
        self.avg      = self.sum / self.count

    def get_val(self):
        return self.this_val
>>>>>>> NewLoader:modules/utils/funtion/averagemeter.py

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')