# -*- coding: utf-8 -*-
# @Time     : 2022/2/5 上午 10:21
# @Author   : zhonzxad
# @File     : user_summary.py
# @desc     : 测试神经网络参数量

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