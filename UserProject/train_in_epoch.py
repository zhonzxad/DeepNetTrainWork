# -*- coding: utf-8 -*-
# @Time     : 2021/12/20 下午 09:26
# @Author   : zhonzxad
# @File     : train_fun.py
import argparse
import os
import sys
import torch
from tqdm import tqdm, trange
from loguru import logger
from torch.autograd import Variable

from UserProject.modules.utils.getloss import loss_func
from UserProject.modules.utils.funtion.averagemeter import AverageMeter

# 在Windows下使用vscode运行时 添加上这句话就会使用正确的相对路径设置
# 需要import os和sys两个库
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def get_lr(optimizer):
    """获取学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_tqdm_post_avg(vals, optimizer):
    """按照固定的顺序对loss相关信息进行输出"""
    names = ["Loss", "CEloss", "BCEloss", "Diceloss", "F_SOCRE", ]
    info = ""
    for i in range(len(vals)):
        if vals[i] > 0:
            info += ("{}={:.5f},".format(names[i], vals[i]))
        elif vals[i] < 0:
            logger.warning("序列化损失函数时发生错误,存在{}损失值小于0的情况".format(names[i]))
        else:
            pass

    info += ("lr={:.7f}".format(get_lr(optimizer)))

    return info

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

def train_in_epoch(net, gens, **kargs):
    """"定义训练每一个epoch的步骤"""
    total_ce_loss   = AverageMeter()
    total_bce_loss  = AverageMeter()
    total_dice_loss = AverageMeter()
    total_f_score   = AverageMeter()
    total_loss      = AverageMeter()

    amp         = kargs["amp"]
    this_device = kargs["device"]
    gpu_ids     = kargs["gpuids"]
    optimizer   = kargs["optimizer"]
    tfwriter    = kargs["tf_writer"]
    cls_weights = kargs["cls_weight"]
    this_epoch  = kargs["this_epoch"]

    # 定义网络为训练模式
    model_train = net.train()

    # 创建混合精度训练
    if amp:
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 分发数据集
    gen_source = gens[0]
    gen_target = gens[1]

    if len(gen_source) == len(gen_target):
        iter_source, iter_target = iter(gen_source), iter(gen_target)
    else:
        iter_source = iter(gen_source)
        iter_target = 0

    tqdm_bar = tqdm(iterable=range(len(gen_source)), leave=False, mininterval=0.2, ascii=True, desc="Train in epoch")
    for batch_idx in tqdm_bar:

        img, png, label = next(iter_source)
        img_tag         = next(iter_target) if iter_target != 0 else None
        # print("\nNo in RangeNet img shape is {} || png shape is {}".format(img.shape, png.shape))

        with torch.no_grad():
            img     = Variable(torch.from_numpy(img).type(torch.FloatTensor), requires_grad=True)
            png     = torch.from_numpy(png).long()
            label   = Variable(torch.from_numpy(label).type(torch.FloatTensor), requires_grad=True)
            weights = torch.from_numpy(cls_weights)
            # logger.write("\n img shape is {} || png shape is {} || seg_labels shape is {}".format(img.shape, png.shape, seg_labels.shape))

            if this_device.type == "cuda":
                img     = img.to(this_device)
                png     = png.to(this_device)
                label   = label.to(this_device)
                weights = weights.to(this_device)

        # 所有梯度为0
        optimizer.zero_grad()

        # 混合精度计算
        if amp:
            with torch.cuda.amp.autocast(enabled=amp):
                # 网络计算
                output = model_train(img)
                # 计算损失
                # print("\n output shape is {} || png shape is {}".format(output.shape, png.shape))
                # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
                loss = loss_func(output, png, label, weights, this_device)

                # 误差反向传播
                # scale作用将梯度进行自动化缩放,它还会判断本轮loss是否是nan，如果是，那么本轮计算的梯度不会回传
                # amp混合精度在2080Ti之后的机型上具有良好的效果，较低配置机型效果不大甚至有nan的风险
                grad_scaler.scale(loss[0]).backward()
                # 优化梯度
                # 首先把梯度的值unscale回来。
                # 如果梯度的值不是 infs 或者 NaNs，那么调用optimizer.step()来更新权重，
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                grad_scaler.step(optimizer)
                # 准备着，看是否要增大scaler
                grad_scaler.update()
        else:
            output = model_train(img)
            # 计算损失 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
            loss   = loss_func(output, png, label, weights, this_device)
            # 损失回传
            loss[0].backward()
            optimizer.step()

        total_loss.update(loss[0].item())
        total_ce_loss.update(loss[1].item())
        total_bce_loss.update(loss[2].item())
        total_dice_loss.update(loss[3].item())
        total_f_score.update(loss[4].item())

        # 写tensorboard
        tags = ["train_loss", "CEloss", "BCEloss", "Diceloss", "f_score", "lr", "accuracy"]
        if tfwriter is not None:
            tfwriter.add_scalar(tags[0], total_loss.get_avg(), this_epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[1], total_ce_loss.get_avg(), this_epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[2], total_bce_loss.get_avg(), this_epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[3], total_dice_loss.get_avg(), this_epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[4], total_f_score.get_avg(), this_epoch*(batch_idx + 1))
            tfwriter.add_scalar(tags[5], get_lr(optimizer), this_epoch*(batch_idx + 1))

        # 设置进度条右边显示的信息
        # tq_str = set_tqdm_post((total_loss, total_ce_loss, total_bce_loss, total_dice_loss, total_f_score),
        #                        (batch_idx + 1), optimizer)
        tq_str = set_tqdm_post_avg((total_loss.get_avg(), total_ce_loss.get_avg(), total_bce_loss.get_avg(),
                                total_dice_loss.get_avg(), total_f_score.get_avg()), optimizer)
        tqdm_bar.set_postfix_str(tq_str)
        # tqdm_bar.update(1)

    # tqdm_bar.close()
    # assert (batch_idx - 1 == total_loss.get_count()), "循环次数与计算损失次数不相等"
    # 返回值按照 0/总loss(float), 1/count(int), 2/[loss], 3/lr
    # loss = tensor格式 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss
    return [total_loss.avg_val(), total_loss.get_count(), loss, get_lr(optimizer)]

def test_in_epoch(net, gen_vals, **kargs):
    """测试方法"""
    total_ce_loss   = AverageMeter()
    total_bce_loss  = AverageMeter()
    total_dice_loss = AverageMeter()
    total_f_score   = AverageMeter()
    total_loss      = AverageMeter()

    amp         = kargs["amp"]
    this_device = kargs["device"]
    gpu_ids     = kargs["gpuids"]
    optimizer   = kargs["optimizer"]
    tfwriter    = kargs["tf_writer"]
    cls_weights = kargs["cls_weight"]
    this_epoch  = kargs["this_epoch"]

    # 设置网络为验证集模式
    model_eval = net.eval()

    # 释放无关内存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # 创建混合精度训练
    if amp:
        grad_scaler_val = torch.cuda.amp.GradScaler(enabled=amp)

    # 分发数据集
    gen_val_source = gen_vals[0]
    gen_val_target = gen_vals[1]

    if len(gen_val_source) == len(gen_val_target):
        iter_source, iter_target = iter(gen_val_source), iter(gen_val_target)
    else:
        iter_source = iter(gen_val_source)
        iter_target = 0

    tqdm_bar_val = tqdm(iterable=range(len(gen_val_source)), leave=False, mininterval=0.2, ascii=True, desc="val")
    for batch_idx_val in tqdm_bar_val:

        img, png, label = next(iter_source)
        img_tag         = next(iter_target) if iter_target != 0 else None

        with torch.no_grad():
            img     = torch.from_numpy(img).type(torch.FloatTensor)
            png     = torch.from_numpy(png).long()
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

        # 在验证和测试阶段不需要计算梯度反向传播
        with torch.no_grad():
            if amp:
                # 混合精度计算
                with torch.cuda.amp.autocast(enabled=amp):
                    # 输入测试图像
                    output    = model_eval(img)

                    # 计算损失
                    # print("\n output shape is {} || png shape is {}".format(output.shape, png.shape))
                    # 返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
                    loss = loss_func(output, png, label, weights, this_device)
                    # grad_scaler_val.scale(loss[0])
                    # # 优化梯度
                    # grad_scaler_val.update()
            else:
                # 输入测试图像
                output = model_eval(img)
                # 计算损失,返回值按照 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss排布
                loss   = loss_func(output, png, label, weights, this_device)

        total_loss.update(loss[0].item())
        total_ce_loss.update(loss[1].item())
        total_bce_loss.update(loss[2].item())
        total_dice_loss.update(loss[3].item())
        total_f_score.update(loss[4].item())

        # 写tensorboard
        tags = ["train_loss_val", "CEloss_val", "BCEloss_val", "Diceloss_val", "f_score_val", "lr_val", "accuracy"]
        if tfwriter is not None:
            tfwriter.add_scalar(tags[0], total_loss.get_avg(), this_epoch*(batch_idx_val + 1))
            tfwriter.add_scalar(tags[1], total_ce_loss.get_avg(), this_epoch*(batch_idx_val + 1))
            tfwriter.add_scalar(tags[2], total_bce_loss.get_avg(), this_epoch*(batch_idx_val + 1))
            tfwriter.add_scalar(tags[3], total_dice_loss.get_avg(), this_epoch*(batch_idx_val + 1))
            tfwriter.add_scalar(tags[4], total_f_score.get_avg(), this_epoch*(batch_idx_val + 1))
            tfwriter.add_scalar(tags[5], get_lr(optimizer), this_epoch*(batch_idx_val + 1))

        #设置进度条右边显示的信息
        # tq_str = set_tqdm_post((total_loss, total_ce_loss, total_bce_loss, total_dice_loss, total_f_score),
        #                        (batch_idx_val + 1), optimizer)
        tq_str_val = set_tqdm_post_avg((total_loss.get_avg(), total_ce_loss.get_avg(), total_bce_loss.get_avg(),
                                    total_dice_loss.get_avg(), total_f_score.get_avg()), optimizer)
        tqdm_bar_val.set_postfix_str(tq_str_val)
        # tqdm_bar_val.update(1)

    # tqdm_bar_val.close()
    # 返回值按照 0/总loss, 1/count, 2/celoss, 3/bceloss, 4/diceloss, 5/floss
    # assert (batch_idx_val - 1 == total_loss.get_count()), "循环次数与计算损失次数不相等"

    # 返回值按照 0/总loss(float), 1/count(int), 2/[loss]
    # loss = tensor格式 0/总loss, 1/celoss, 2/bceloss, 3/diceloss, 4/floss
    return [total_loss.avg_val(), total_loss.get_count(), loss]