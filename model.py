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

logger.add('run.log')

def knn(x, k):
    # todo: 1. 修改距离公式, 2. normalization。解决办法：新加一个全连接层
    # multiply feature vector pointwisely
    inner = -2 * torch.matmul(x.transpose(2, 1),
                              x)  # (batch_size, feature_size, num_points) -> (batch_size, num_points, num_points)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (batch_size, 1, num_points)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

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
        self.dense1 = nn.Linear(input_size, 256)
        self.dense2 = nn.Linear(256, 18)
        self.dense3 = nn.Linear(18, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return x


class DGCNN_semseg(nn.Module):
    # Todo: train model on parallel, ref: https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/12
    def __init__(self, args, input_feature_size=3):
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

        self.dense1 = nn.Linear(input_feature_size, input_feature_size)

    def forward(self, x):
        num_points = x.size(1)
        x = F.relu(self.dense1(x))
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k,
                              dim9=False)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
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
        x = self.dp1(x)
        # x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x


class DataParallel_wrapper(nn.Module):
    # Todo: 解决loss随着卡数增多而增大的问题
    def __init__(self, model, row_classifier, col_classifier):
        super(DataParallel_wrapper, self).__init__()
        self.model = model
        self.row_classifier, self.col_classifier = row_classifier, col_classifier

    def forward(self, data, row_matrix, col_matrix, num_vertices):

        all_features = self.model(data)  # (batch_size, 256, max_num_vertices)
        batch_size = all_features.size(0)
        # check whether generated features equals to zero vector
        with torch.no_grad():
            for i in range(batch_size):
                zero_count = torch.sum(all_features[i, :, :num_vertices[i]], dim=1)  # (num_vertices[i])
                features_all_zero = torch.nonzero(zero_count == 0) # (zero_count * 1)
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

        # row_point3 = torch.zeros([row_input_shape, 256])
        # col_point3 = torch.zeros([col_input_shape, 256])
        # get negative sampling

        for i, v in enumerate(num_vertices):
            row_batch_idx = (zero_row[:, 0] == i)
            col_batch_idx = (zero_col[:, 0] == i)
            small_than_row = torch.logical_and((zero_row[row_batch_idx][:, 1] < v),
                                               (zero_row[row_batch_idx][:, 2] < v))
            small_than_col = torch.logical_and((zero_col[col_batch_idx][:, 1] < v),
                                               (zero_col[col_batch_idx][:, 2] < v))

            point3 = all_features[zero_row[row_batch_idx][small_than_row][:, 0], :,
                     zero_row[row_batch_idx][small_than_row][:, 1]]
            # point4 = all_features[zero_row[row_batch_idx][small_than_row][:, 0], :, zero_row[row_batch_idx][small_than_row][:, 2]]

            point5 = all_features[zero_col[col_batch_idx][small_than_col][:, 0], :,
                     zero_col[col_batch_idx][small_than_col][:, 1]]
            # point6 = all_features[zero_col[col_batch_idx][small_than_col][:, 0], :, zero_col[col_batch_idx][small_than_col][:, 2]]
            if i == 0:
                row_point3 = point3
                # row_point4 = point4
                col_point3 = point5
                # col_point4 = point6
            else:
                row_point3 = torch.cat((row_point3, point3), dim=0)
                # row_point4 = torch.cat((row_point4, point4), dim=0)
                col_point3 = torch.cat((col_point3, point5), dim=0)
                # col_point4 = torch.cat((col_point4, point6), dim=0)

        row_zero_shape = row_point3.size(0)
        col_zero_shape = col_point3.size(0)
        # col_point3 = all_features[zero_col[:, 0], :, zero_col[:, 1]]
        # col_point4 = all_features[zero_col[:, 0], :, zero_col[:, 2]]
        row_weights = torch.ones([row_zero_shape])
        col_weights = torch.ones([col_zero_shape])

        logger.debug(f'shape of row_weights: {row_weights.shape}')
        logger.debug(f'shape of row_input_shape: {row_input_shape}')
        # sampling negative samples from original matrix
        row_sampling = torch.multinomial(row_weights,
                                         row_input_shape) if row_zero_shape > row_input_shape else torch.arange(
            row_zero_shape)
        col_sampling = torch.multinomial(col_weights,
                                         col_input_shape) if col_zero_shape > col_input_shape else torch.arange(
            col_zero_shape)

        logger.debug(f'row sampling shape: {row_sampling.shape}')
        logger.debug(f'row point3 shape: {row_point3.shape}')

        # when #negative sampling examples is smaller than positive number, we need to expand negative example
        # a sample strategy is to padding negative vector with zeros

        row_neg_point = torch.zeros([row_input_shape, 256]).to('cuda')
        if row_input_shape > row_sampling.size(0):
            logger.warning(f'row negative sampling padding with zero vector')
        row_neg_point[:row_sampling.size(0)] = row_point3[row_sampling]

        # row_point4 = row_point4[:row_input_shape] if row_point4.size(0) > (row_input_shape) else row_point4
        col_neg_point = torch.zeros([col_input_shape, 256]).to('cuda')
        if col_input_shape > col_sampling.size(0):
            logger.warning(f'col negative sampling padding with zero vector')
        col_neg_point[:col_sampling.size(0)] = col_point3[col_sampling]
        # col_point4 = col_point4[:col_input_shape] if col_point4.size(0) > (col_input_shape) else col_point4

        # predict relation between row point & col point

        row_pos_input = torch.cat((row_point1, row_point2), 1)

        row_neg_input1, row_neg_input2 = torch.cat((row_neg_point, row_point1), 1), torch.cat(
            (row_neg_point, row_point2),
            1)
        row_neg_input = torch.cat((row_neg_input1, row_neg_input2), 0)

        col_pos_input = torch.cat((col_point1, col_point2), 1)
        col_neg_input1, col_neg_input2 = torch.cat((col_neg_point, col_point1), 1), torch.cat(
            (col_neg_point, col_point2),
            1)
        col_neg_input = torch.cat((col_neg_input1, col_neg_input2), 0)
        row_input = torch.cat((row_pos_input, row_neg_input), 0)  # (batch_size, *, 512)
        col_input = torch.cat((col_pos_input, col_neg_input), 0)  # (batch_size, *, 512)

        row_pos_gt, row_neg_gt = torch.ones([row_input_shape], dtype=torch.long), torch.zeros(
            [row_neg_input.size(0)],
            dtype=torch.long)
        col_pos_gt, col_neg_gt = torch.ones([col_input_shape], dtype=torch.long), torch.zeros(
            [col_neg_input.size(0)],
            dtype=torch.long)

        row_pos_gt, row_neg_gt = row_pos_gt.to('cuda'), row_neg_gt.to('cuda')
        col_pos_gt, col_neg_gt = col_pos_gt.to('cuda'), col_neg_gt.to('cuda')
        row_gt = torch.cat((row_pos_gt, row_neg_gt), 0)
        col_gt = torch.cat((col_pos_gt, col_neg_gt), 0)

        row_shuffle, col_shuffle = torch.randperm(row_gt.size(0)), torch.randperm(col_gt.size(0))
        row_input, col_input = row_input[row_shuffle], col_input[col_shuffle]
        row_gt, col_gt = row_gt[row_shuffle], col_gt[col_shuffle]

        row_output = F.softmax(self.row_classifier(row_input), 1)
        col_output = F.softmax(self.col_classifier(col_input), 1)

        loss = F.cross_entropy(row_output, row_gt) + F.cross_entropy(col_output, col_gt)
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
