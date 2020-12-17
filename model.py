#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from loguru import logger
import itertools


def sampling(matrix: torch.Tensor, sampling_num: int):
    '''
    sampling negative examples from input matrix
    :param matrix: the input matrix, (batch_size, num_vertice, num_vertice)
    :param sampling_num: the totally sampling number, -1 for all sample
    :return: the idx of negative pairs (zeros, 2)
    '''
    matrix_size = matrix.size(1)
    idx_production = torch.tensor(list(itertools.product(range(matrix_size), range(matrix_size))), dtype=torch.long).to('cuda')
    # matrix with batch size equals to 1
    matrix = matrix.squeeze(dim=0)
    zero_idx = (matrix[idx_production[:,0],idx_production[:,1]] == 0)
    idx_production = idx_production[zero_idx] # (zero, 2)
    if sampling_num == -1:
        return idx_production
    # sampling from idx_production with sampling_num
    weights = torch.ones(idx_production.size(0)).to('cuda')
    sampling_idx = torch.arange(idx_production.size(0)) if idx_production.size(0) < sampling_num else torch.multinomial(weights, sampling_num)
    idx_production = idx_production[sampling_idx]
    return idx_production

def knn(x, k, dim_expand=False):
    # multiply feature vector pointwisely
    # x: (batch_size, feature_size, num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    device = torch.device('cuda')
    if dim_expand:
        if k > num_points:
            incr_list = torch.arange(num_points, device=device, dtype=torch.int)
            ret = torch.zeros([batch_size, num_points, num_points], device=device, dtype=torch.int)
            ret[:, :] = incr_list
            return ret
        x1, y1 = x[:, 0, :].unsqueeze(2), x[:, 2, :].unsqueeze(2)  # (batch_size, num_points, 1)
        dis_x, dis_y = x1.transpose(2, 1), y1.transpose(2, 1)
        x1, y1 = x1.expand(batch_size, num_points, num_points), y1.expand(batch_size, num_points, num_points)
        pairwise_x, pairwise_y = (x1 - dis_x).abs(), (y1 - dis_y).abs()
        pairwise_distance = pairwise_x + pairwise_y
        # idx_x, idx_y = torch.topk(pairwise_x, k=k, dim=-1, largest=False)[1], \
        #                torch.topk(pairwise_y, k=k, dim=-1, largest=False)[1]
        idx = torch.topk(pairwise_distance, k=k, dim=-1, largest=False)[1]
        # idx = torch.cat((idx_x, idx_y), dim=-1)
        return idx

    else:
        if k > num_points:
            incr_list = torch.arange(num_points, device=device, dtype=torch.int)
            ret = torch.zeros([batch_size, num_points, num_points], device=device, dtype=torch.int)
            ret[:, :] = incr_list
            return ret
        inner = -2 * torch.matmul(x.transpose(2, 1),
                                  x)  # (batch_size, feature_size, num_points) -> (batch_size, num_points, num_points)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (batch_size, 1, num_points)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx


@logger.catch
def get_graph_feature(x, k=40, idx=None, dim_expand=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not dim_expand:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, :4], k=k, dim_expand=True)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx.to(device)
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, -1, num_dims)

    actual_k = feature.size(2)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, actual_k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        # x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class PairClassfier(nn.Module):
    def __init__(self, input_size):
        super(PairClassfier, self).__init__()
        self.dp1 = nn.Dropout()
        self.dense1 = nn.Linear(input_size, 256)
        self.dense2 = nn.Linear(256, 18)
        self.dense3 = nn.Linear(18, 2)

    def forward(self, x):
        logger.debug(f'input x: {x}')
        # x = self.dp1(x)
        x = F.relu(self.dense1(x))
        x = self.dp1(x)
        x = F.relu(self.dense2(x))
        x = self.dp1(x)
        x = self.dense3(x)
        return x


class DGCNN_semseg(nn.Module):
    # Todo: train model on parallel, ref: https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/12
    def __init__(self, args, input_feature_size=14):
        super(DGCNN_semseg, self).__init__()
        self.args = args
        self.k = args.k
        self.input_feature_size = input_feature_size
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(self.input_feature_size * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

        # self.dense1 = nn.Linear(input_feature_size, input_feature_size)

    def forward(self, x):
        num_points = x.size(1)
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k,
                              dim_expand=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x


class DataParallel_wrapper(nn.Module):
    def __init__(self, args, model, row_classifier, col_classifier):
        super(DataParallel_wrapper, self).__init__()
        self.model = model
        self.row_classifier, self.col_classifier = row_classifier, col_classifier
        self.args = args

    def forward(self, data, row_matrix, col_matrix, num_vertices):
        # Todo: 检查column关系准确率低的问题
        if self.args.dgcnn:
            all_features = self.model(data)  # (batch_size, 256, num_vertices)
        else:
            all_features = data[:,:,:4].permute(0,2,1) # (batch_size, 4, num_vertices)
        logger.debug(f'all feature shape: {all_features.shape}')
        # logger.debug(f'all features: {all_features.permute(0,2,1)}')
        batch_size = all_features.size(0)
        logger.debug(f'num vertices: {num_vertices[0]}')
        # check whether generated features equals to zero vector
        with torch.no_grad():
            for i in range(batch_size):
                zero_count = torch.sum(all_features[i, :, :num_vertices[i]], dim=1)  # (num_vertices[i])
                features_all_zero = torch.nonzero(zero_count == 0)  # (zero_count * 1)
                if features_all_zero.size(0) != 0:
                    logger.warning(f'the features of points: {features_all_zero[:, 0]} is all zeros!')

        non_zero_row = torch.nonzero(row_matrix)
        non_zero_col = torch.nonzero(col_matrix)
        zero_row = torch.nonzero((row_matrix == 0))
        zero_col = torch.nonzero((col_matrix == 0))
        row_point1 = all_features[non_zero_row[:, 0], :, non_zero_row[:, 1]]  # (non_zero, 256)
        row_point2 = all_features[non_zero_row[:, 0], :, non_zero_row[:, 2]]
        row_input_shape = row_point1.size(0)

        col_point1 = all_features[non_zero_col[:, 0], :, non_zero_col[:, 1]]
        col_point2 = all_features[non_zero_col[:, 0], :, non_zero_col[:, 2]]
        col_input_shape = col_point1.size(0)

        logger.debug(f'row point1 shape: {row_point1.shape}\tcol point1 shape {col_point1.shape}')
        col_neg_idx_pairs = sampling(col_matrix, -1)
        row_neg_idx_pairs = sampling(row_matrix, -1)

        if self.args.dgcnn:
            row_pos_input = torch.cat((row_point1, row_point2), 1)
            row_neg_input = torch.cat((all_features[:, :, row_neg_idx_pairs[:, 0]], all_features[:, :, row_neg_idx_pairs[:, 1]]), dim=1)
            row_neg_input = row_neg_input.squeeze().transpose(1,0)
            col_pos_input = torch.cat((col_point1, col_point2), 1)
            col_neg_input = torch.cat((all_features[:, :, col_neg_idx_pairs[:, 0]], all_features[:, :, col_neg_idx_pairs[:, 1]]), dim=1)
            col_neg_input = col_neg_input.squeeze().transpose(1, 0)
            logger.debug(f'row_pos_input shape: {row_pos_input.shape}\trow_neg_input shape: {row_neg_input.shape}')

        else:
            row_pos_input = row_point1 - row_point2
            col_pos_input = col_point1 - col_point2
            row_neg_input = all_features[:, :, row_neg_idx_pairs[:, 0]].squeeze().transpose(1,0) - all_features[:, :, row_neg_idx_pairs[:, 1]].squeeze().transpose(1,0)
            col_neg_input = all_features[:, :, col_neg_idx_pairs[:, 0]].squeeze().transpose(1,0) - all_features[:, :, col_neg_idx_pairs[:, 1]].squeeze().transpose(1,0)

        row_input = torch.cat((row_pos_input, row_neg_input), 0)  # (batch_size, *, 512)
        col_input = torch.cat((col_pos_input, col_neg_input), 0)  # (batch_size, *, 512)
        row_pos_gt, row_neg_gt = torch.ones([row_pos_input.size(0)], dtype=torch.long), torch.zeros(
            [row_neg_input.size(0)],
            dtype=torch.long)
        col_pos_gt, col_neg_gt = torch.ones([col_pos_input.size(0)], dtype=torch.long), torch.zeros(
            [col_neg_input.size(0)],
            dtype=torch.long)

        row_pos_gt, row_neg_gt = row_pos_gt.to('cuda'), row_neg_gt.to('cuda')
        col_pos_gt, col_neg_gt = col_pos_gt.to('cuda'), col_neg_gt.to('cuda')
        row_gt = torch.cat((row_pos_gt, row_neg_gt), 0)
        col_gt = torch.cat((col_pos_gt, col_neg_gt), 0)
        assert row_input.size(0) == row_gt.size(
            0), f'row input size {row_input.shape} different from row gt size {row_gt.shape}'
        assert col_input.size(0) == col_gt.size(
            0), f'col input size {col_input.shape} different from col gt size {col_gt.shape}'

        logger.debug(f'row input shape: {row_input.shape}\trow gt shape: {row_gt.shape}')
        logger.debug(f'col input shape: {col_input.shape}\tcol gt shape: {col_gt.shape}')
        row_shuffle, col_shuffle = torch.randperm(row_gt.size(0)), torch.randperm(col_gt.size(0))
        logger.debug(f'row shuffle: {row_shuffle.shape}\tcol shuffle: {col_shuffle.shape}')
        row_input, col_input = row_input[row_shuffle], col_input[col_shuffle]
        row_gt, col_gt = row_gt[row_shuffle], col_gt[col_shuffle]
        logger.debug(f'row gt: {row_gt}\ncol gt: {col_gt}')
        row_output = self.row_classifier(row_input)
        col_output = self.col_classifier(col_input)
        # row_output = F.softmax(self.row_classifier(row_input), 1)
        # col_output = F.softmax(self.col_classifier(col_input), 1)
        loss_weights = torch.tensor([0.2, 1.0]).to('cuda')
        row_loss = F.cross_entropy(row_output, row_gt, weight=loss_weights) if row_output.size(0) > 0 else 0
        col_loss = F.cross_entropy(col_output, col_gt, weight=loss_weights) if col_output.size(0) > 0 else 0
        loss = row_loss + col_loss
        # logger.info(f'loss shape: {loss.shape}\tloss value: {loss}')
        # loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss, 0)


# def DataParallel_withLoss(model, loss, **kwargs):
#     model = DataParallel_wrapper(model, loss)
#     if 'device_ids' in kwargs.keys():
#         device_ids = kwargs['device_ids']
#     else:
#         device_ids = None
#     if 'output_device' in kwargs.keys():
#         output_device = kwargs['output_device']
#     else:
#         output_device = None
#     if 'cuda' in kwargs.keys():
#         cudaID = kwargs['cuda']
#         model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
#     else:
#         model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
#     return model

if __name__ == '__main__':
    a = torch.arange(25).view(1,5,5)
    a[0,2,:] = 0
    print(sampling(a, 5))
