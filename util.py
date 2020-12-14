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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

from loguru import logger
from matplotlib import collections as mc


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
    logger.debug(f'pred matrix shape: {pred.shape}\tgt matrix shape: {gt.shape}')
    num_vertices = pred.size(0)
    logger.debug(pred)
    logger.debug(gt)
    TP = torch.nonzero((pred - gt) == 0).size(0)
    FP = num_vertices * num_vertices - TP
    binary_TP = torch.sum(torch.logical_and(pred, gt)).item()
    # False negative means the relation in gt is positive while in predicted is negative
    # False positive means the relation in gt is negative while in predicted is positive
    pred_gt = pred - gt
    binary_FP = torch.nonzero(pred_gt == 1).size(0)
    binary_FN = torch.nonzero(pred_gt == -1).size(0)
    # all_ones = torch.ones([num_vertices, num_vertices], dtype=torch.int).to('cuda')
    # tmp = gt - pred - all_ones
    # FN = torch.nonzero(tmp==0).size(0)
    # correct_relations = torch.sum(torch.logical_and(pred, gt))
    # exists_relations = torch.nonzero(gt).size(0)
    if TP == 0 and FP == 0:
        logger.warning(f'TP and FP is both zero')
        precision = 0
    else:
        precision = TP / (TP + FP)
    if binary_TP == 0 and binary_FP == 0:
        logger.warning(f'[b] TP and [b] FP is both zero')
        binary_precision = 0
    else:
        binary_precision = binary_TP / (binary_TP + binary_FP)
    if binary_TP == 0 and binary_FN == 0:
        logger.warning(f'[b] TP and [b] FN is both zero')
        binary_recall = 0
    else:
        binary_recall = binary_TP / (binary_TP + binary_FN)
    # recall = TP / (TP+FN)
    return TP, FP, precision, binary_TP, binary_FP, binary_FN, binary_precision, binary_recall


def error_correction(pred_matrix, classifier_output):
    """
    Error correction for pred_matrix based on classifier output (set the pred_matrix to be symmetric and elements along diagonal to be zero)
    :param pred_matrix: the predicated matrix
    :param classifier_output: output of classifier
    :return: the corrected matrix
    """
    # todo: set diagonal elements value with zero
    # todo: set pred_matrix to be symmetric
    classifier_output = classifier_output.squeeze()
    num_vertices = pred_matrix.size(0)
    is_symmetric = torch.nonzero((pred_matrix == pred_matrix.t()) == False)
    if is_symmetric.size(0) != 0:
        logger.warning(f'The pred matrix is not symmetric!')
        classifier_output = F.softmax(classifier_output, dim=-1).view(num_vertices, num_vertices,
                                                                      2)  # (num_vertices, num_vertices, 2)
        classifier_output, idx = torch.max(classifier_output, dim=-1)  # (num_vertices, num_vertices)
        classifier_output = classifier_output - classifier_output.t()  # (num_vertices, num_vertices)
        # todo: 处理置信度相等而类型不同的情况
        final_result = torch.nonzero(classifier_output > 0)  # (non_zero, 2)
        pred_matrix[final_result[:, 0], final_result[:, 1]] = idx[final_result[:, 0], final_result[:, 1]].type(
            torch.IntTensor).to('cuda')
        pred_matrix[final_result[:, 1], final_result[:, 0]] = idx[final_result[:, 0], final_result[:, 1]].type(
            torch.IntTensor).to('cuda')
    diagonal_zero = torch.nonzero(torch.diag(torch.diagonal(pred_matrix)))
    if diagonal_zero.size(0) != 0:
        logger.warning(f'Non zero elements in diagonal!')
        pred_matrix[diagonal_zero[:, 0], diagonal_zero[:, 1]] = 0
    return pred_matrix


def draw_figure(num_vertices, rbp, rbr, cbp, cbr):
    rbp, rbr, cbp, cbr = np.array(rbp), np.array(rbr), np.array(cbp), np.array(cbr)
    rbf = (2 * rbp * rbr) / (rbp + rbr)
    cbf = (2 * cbp * cbr) / (cbp + cbr)
    plt.clf()
    plt.figure()
    plt.subplot(231)
    plt.scatter(num_vertices, rbp, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('row precision')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    # plt.title('[rp]#vertices')

    # log
    plt.subplot(232)
    plt.scatter(num_vertices, rbr, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('row recall')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    # plt.title('[rr]#vertices')
    plt.subplot(233)
    plt.scatter(num_vertices, rbf, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('row f1')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    # symmetric log
    plt.subplot(234)
    plt.scatter(num_vertices, cbp, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('column precision')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    # plt.title('[cp]#vertices')

    # logit
    plt.subplot(235)
    plt.scatter(num_vertices, cbr, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('column recall')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)

    plt.subplot(236)
    plt.scatter(num_vertices, cbf, s=1, alpha=0.1)
    plt.xlabel('#vertices')
    plt.ylabel('column f1')
    plt.xticks(np.arange(min(num_vertices), max(num_vertices) + 1, 100.0), rotation=60, horizontalalignment='right')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    # plt.title('[cr]#vertices')
    plt.tight_layout()

    return plt


def get_center_of_chunks(x1, x2, y1, y2):
    """
    get center position of a chunk
    """
    width, height = x2 - x1, y2 - y1
    x_center, y_center = (x1 + width / 2), (y1 + height / 2)
    return x_center, y_center


def draw_rectangle(pred_matrix, gt_matrix, chunks):
    """
    draw rectangle to show the difference between prediction and the ground truth
    :param pred_matrix: the predicted matrix (torch.tensor)
    :param gt_matrix: the ground truth matrix (torch.tensor)
    :param chunks_path: chunks file path with a list of chunk in the scitsr dataset
    :return: a plt figure
    """
    figure, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    records = np.zeros([len(chunks), 4])
    for i, chunk in enumerate(chunks):
        pos = chunk['pos']
        x1, x2, y1, y2 = pos[0], pos[1], pos[2], pos[3]
        records[i] = x1, x2, y1, y2
        width, height = x2 - x1, y2 - y1
        rect1 = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='b', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='b', facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)
    # adding line based on pred_matrix and ground truth matrix
    x_center = (records[:, 1] - records[:, 0]) / 2 + records[:, 0]
    y_center = (records[:, 3] - records[:, 2]) / 2 + records[:, 2]
    pred_idx, gt_idx = torch.nonzero(pred_matrix), torch.nonzero(gt_matrix)
    pred_related, gt_related = pred_idx.cpu().numpy(), gt_idx.cpu().numpy()
    pred_start, pred_end, gt_start, gt_end = pred_related[:, 0], pred_related[:, 1], gt_related[:, 0], gt_related[:,
                                                                                                       1]  # (non_zero, 1)
    pred_start_x, pred_start_y, pred_end_x, pred_end_y = x_center[pred_start], y_center[pred_start], x_center[pred_end], \
                                                         y_center[pred_end]
    gt_start_x, gt_start_y, gt_end_x, gt_end_y = x_center[gt_start], y_center[gt_start], x_center[gt_end], y_center[
        gt_end]
    pred_start_x, pred_start_y, pred_end_x, pred_end_y = np.expand_dims(pred_start_x, axis=1), np.expand_dims(
        pred_start_y, axis=1), np.expand_dims(pred_end_x, axis=1), np.expand_dims(pred_end_y, axis=1)
    gt_start_x, gt_start_y, gt_end_x, gt_end_y = np.expand_dims(gt_start_x, axis=1), np.expand_dims(gt_start_y,
                                                                                                    axis=1), np.expand_dims(
        gt_end_x, axis=1), np.expand_dims(gt_end_y, axis=1)
    logger.info(f'pred_start_x shape: {pred_start_x.shape}\tpred_end_x_shape: {pred_end_x.shape}')
    logger.info(f'pred_start_y shape: {pred_start_y.shape}\tpred_end_y_shape: {pred_end_y.shape}')
    pred_start, pred_end = np.concatenate((pred_start_x, pred_start_y), axis=1), np.concatenate(
        (pred_end_x, pred_end_y), axis=1)
    gt_start, gt_end = np.concatenate((gt_start_x, gt_start_y), axis=1), np.concatenate((gt_end_x, gt_end_y), axis=1)
    pred_lines, gt_lines = list(zip(pred_start, pred_end)), list(zip(gt_start, gt_end))
    pred_lc, gt_lc = mc.LineCollection(pred_lines), mc.LineCollection(gt_lines)
    ax1.add_collection(pred_lc)
    ax1.title.set_text('predicted')
    ax2.add_collection(gt_lc)
    ax2.title.set_text('ground truth')
    ax1.autoscale()
    ax2.autoscale()
    return figure


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
    # a = torch.ones([3, 3])
    # b = torch.ones([3, 3])
    # c = torch.zeros([3, 3])
    # c[0][0] = 1
    # logger.info(metric(a, b))
    # logger.info(metric(a, c))
    # a = torch.arange(25, dtype=torch.int).view(5, 5).to('cuda')
    # b = torch.rand([5, 5, 2]).to('cuda')
    # print(error_correction(a, b))
    # a = list(range(6))
    # b = c = d = e = a
    # plt = draw_figure(a, b, c, d, e)
    # plt.savefig('test')
    import json
    a, b = torch.ones([10, 10]), torch.zeros([10, 10])
    chunk_path = '/home/lihuichao/academic/SciTSR/dataset/test/chunk/0001020v1.12.chunk'
    with open(chunk_path, 'r') as f:
        obj = json.load(f)
    chunks = obj['chunks']
    fig = draw_rectangle(a, b, chunks)
    fig.savefig('rec_test')
