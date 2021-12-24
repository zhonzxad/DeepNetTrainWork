import math

import numpy as np
import torch


def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]  * pred[i,:,:,:])
        Ior1  = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:]) - Iand1
        IoU1  = Iand1 / Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

def PA(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target
 
    return (torch.sum(tmp).float() / input.nelement())


def mIOU(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    # 创建[b,c,h,w]大小的0矩阵
    inputTmp  = torch.zeros([input.shape[0], classNum,input.shape[1],input.shape[2]])
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])
    # 将input维度扩充为[b,1,h,w]
    input  = input.unsqueeze(1)
    target = target.unsqueeze(1)
    # input作为索引，将0矩阵转换为onehot矩阵
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)
    # 为该batch中每张图像存储一个miou
    batchMious = []
    # 乘法计算后，其中1的个数为intersection
    mul = inputOht * targetOht

    # 遍历图像
    for i in range(input.shape[0]):
        ious = []
        # 遍历类别，包括背景
        for j in range(classNum):
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
            
        #计算该图像的miou
        miou = np.mean(ious)
        batchMious.append(miou)

    return batchMious

def IOU_loss(pred, label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious
