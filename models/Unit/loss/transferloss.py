'''
Author: zhonzxad
Date: 2021-12-01 15:27:02
LastEditTime: 2021-12-01 15:34:39
LastEditors: zhonzxad
'''
import torch
import torch.nn as nn

from loss.TransferLossFun import *

"""
kwargs
transfer_loss_args = {
    "loss_type": self.transfer_loss,
    "max_iter": max_iter,
    "num_class": num_class
}
"""

class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        if loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "lmmd":
            self.loss_func = LMMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "adv":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == "daan":
            self.loss_func = DAANLoss(**kwargs)
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)
