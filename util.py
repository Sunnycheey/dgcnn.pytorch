#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def metric(pred: torch.Tensor, gt: torch.Tensor):
    """
    Calculate metric (TP, FP, FN, micro precision/recall, macro precision/recall)
    :param pred: the predicted matrix (num_vertices, num_vertices)
    :param gt: the ground truth matrix
    :return: metric
    """
    logger.info(f'pred matrix shape: {pred.shape}\tgt matrix shape: {gt.shape}')
    num_vertices = pred.size(0)
    # TP = torch.sum(torch.eq(pred, gt)).item()
    logger.info(pred)
    logger.info(gt)
    TP = torch.sum(torch.logical_and(pred, gt)).item()
    FP = torch.sum(pred).item() - TP
    # False negative means the relation in gt is positive while in predicted is negative
    all_ones = torch.ones([num_vertices, num_vertices], dtype=torch.int).to('cuda')
    tmp = gt - pred - all_ones
    FN = torch.nonzero(tmp==0).size(0)
    correct_relations = torch.sum(torch.logical_and(pred, gt))
    exists_relations = torch.nonzero(gt).size(0)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    return TP, FP, FN, precision, recall, correct_relations, exists_relations


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

if __name__ == '__main__':
    a = torch.ones([3,3],dtype=torch.int).to('cuda')
    b = torch.ones([3,3],dtype=torch.int).to('cuda')
    print(metric(a,b))
