# -*- coding: utf-8 -*-
# @Time     : 2021/12/2 下午 12:02
# @Author   : zhonzxad
# @File     : __init__.py

__all__ = ["MMDLoss", "LMMDLoss", "CORAL", "AdversarialLoss", "DAANLoss", ]
from TransferLossFun.mmd import *
from TransferLossFun.adv import *
from TransferLossFun.coral import *
from TransferLossFun.daan import *
from TransferLossFun.lmmd import *

