#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg.py
@Time: 2020/2/24 7:17 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS, SciTSR
from model import DGCNN_semseg, PairClassfier, DataParallel_wrapper
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, metric, error_correction, draw_figure, draw_rectangle
import sklearn.metrics as metrics
import shutil
import json
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm



def _init_():
    # todo: 模型迁移至cpu
    # todo: 训练完一段时间之后进行验证
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not os.path.exists(args.saved_model_dir):
        os.makedirs(args.saved_model_dir)
    os.system('cp main_semseg.py checkpoints' + '/' + args.exp_name + '/' + 'main_semseg.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    if args.log:
        logger.add(args.log)

def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def train(args, io):
    padding = True
    if args.batch_size == 1:
        padding = False
    train_loader = DataLoader(SciTSR(partition=args.train_partition, dataset_dir='/home/lihuichao/academic/SciTSR/dataset', normalize=True, chunk=args.chunk, rel=args.rel),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(SciTSR(partition='test', dataset_dir='/home/lihuichao/academic/SciTSR/dataset', normalize=True),
    #                          num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args, 14).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    if args.clear:
        shutil.rmtree(args.saved_model_dir)
        os.makedirs(args.saved_model_dir)
    writer = None
    if args.summary:
        # os.makedirs(args.summary)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.summary)
    if args.dgcnn:
        row_classifier, col_classifier = PairClassfier(512), PairClassfier(512)
    else:
        row_classifier, col_classifier = PairClassfier(4), PairClassfier(4)
    # row_classifier, col_classifier = nn.DataParallel(row_classifier), nn.DataParallel(col_classifier)
    # model = nn.DataParallel(model)
    model.train()
    row_classifier.train()
    col_classifier.train()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    params = list(model.parameters()) + list(row_classifier.parameters()) + list(col_classifier.parameters())

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(params, lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(params, lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)
    iterate_step = 0

    if args.pretrained_model:
        model_path = args.pretrained_model
        iterate_step = int(model_path.split('_')[-1])
        logger.info(f'start load weights from pretrained model....')
        model_dict = torch.load(args.pretrained_model)
        model.load_state_dict(model_dict['dgcnn'])
        # row_classifier, col_classifier = nn.DataParallel(row_classifier), nn.DataParallel(col_classifier)
        row_classifier.load_state_dict(model_dict['row_classifier'])
        col_classifier.load_state_dict(model_dict['col_classifier'])
        logger.info(f'load successfully!')

    # best_test_iou = 0

    data_paraller = DataParallel_wrapper(args, model, row_classifier, col_classifier)
    data_paraller.to(device)
    data_paraller = nn.DataParallel(data_paraller)
    inner_count = 0
    data_paraller.train()
    for epoch in tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        # model.train()
        # row_classifier.train()
        # col_classifier.train()
        for data in tqdm(train_loader):
            with logger.catch():
                if iterate_step != 0 and inner_count <= iterate_step:
                    inner_count += 1
                    continue
                iterate_step += 1
                file, features, row_matrix, col_matrix, num_vertices = data['file_path'], data['features'].type(torch.FloatTensor), data[
                    'row_matrix'], data['col_matrix'], data['num_vertices']
                # logger.debug(f'input row matrix: {row_matrix}\ninput col matrix: {col_matrix}')
                logger.debug(f'input features: {features}')
                batch_size = features.size()[0]
                opt.zero_grad()
                loss = data_paraller(features, row_matrix, col_matrix, num_vertices)
                loss = loss.sum()

                if writer:
                    writer.add_scalar('Loss/expand_rel_linear', loss.item(), iterate_step)
                    logger.info(f'loss value: {loss.item()}')
                loss.backward()
                opt.step()
                # pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
                count += batch_size
                train_loss += loss.item() * batch_size
                torch.cuda.empty_cache()
                if iterate_step % args.save_step == 0:
                    torch.save({'dgcnn': model.state_dict(), 'row_classifier': row_classifier.state_dict(),
                                'col_classifier': col_classifier.state_dict()},
                               f'{args.saved_model_dir}/checkpoints_{iterate_step}')
                    if writer:
                        writer.flush()
        torch.save({'dgcnn': model.state_dict(), 'row_classifier': row_classifier.state_dict(),
                    'col_classifier': col_classifier.state_dict()},
                   f'{args.saved_model_dir}/checkpoints_{iterate_step}')
        if writer:
            writer.flush()
        # outstr = 'Train %d, loss: %.6f ' % (epoch, train_loss * 1.0 / count)
        # io.cprint(outstr)


def test(args, io):

    if args.summary:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.summary)

    test_loader = DataLoader(SciTSR(partition='test', dataset_dir='/home/lihuichao/academic/SciTSR/dataset', normalize=True),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        device = torch.device("cuda" if args.cuda else "cpu")
        dgcnn = DGCNN_semseg(args, 14).to(device)
        # dgcnn = nn.DataParallel(dgcnn)
        state_dict = torch.load(args.model_path)
        logger.info(state_dict.keys())
        # create new OrderedDict that does not contain `module.`
        model_dict = state_dict
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`
        #     model_dict[name] = v
        # logger.info(model_dict.keys())
        dgcnn.load_state_dict(model_dict['dgcnn'])
        if args.dgcnn:
            row_classifier, col_classifier = PairClassfier(512), PairClassfier(512)
        else:
            row_classifier, col_classifier = PairClassfier(4), PairClassfier(4)

        # row_classifier, col_classifier = nn.DataParallel(row_classifier), nn.DataParallel(col_classifier)
        row_classifier.load_state_dict(model_dict['row_classifier'])
        col_classifier.load_state_dict(model_dict['col_classifier'])
        print(row_classifier)
        dgcnn, row_classifier, col_classifier = dgcnn.to(device), row_classifier.to(device), col_classifier.to(device)
        dgcnn, row_classifier, col_classifier = dgcnn.eval(), row_classifier.eval(), col_classifier.eval()
        count = 0
        row_TP, row_FP, b_row_TP, b_row_FP, b_row_FN, col_TP, col_FP, b_col_TP, b_col_FP, b_col_FN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        row_macro_precision, col_macro_precision = [], []
        b_row_macro_precision, b_row_macro_recall, b_col_macro_precision, b_col_macro_recall = [], [], [], []
        num_vertices_list = []
        for data in tqdm(test_loader):
            with logger.catch():
                file_path, features, row_matrix, col_matrix, num_vertices = data['file_path'], data['features'].type(torch.FloatTensor), data[
                    'row_matrix'], data['col_matrix'], data['num_vertices']
                data, row_matrix, col_matrix, num_vertices = features.to(device), row_matrix.to(device), col_matrix.to(
                    device), num_vertices.to(device)
                logger.info(f'file path: {file_path}')
                logger.debug(
                    f'num_vertices value: {num_vertices[0].item()}\tdata shape: {data.shape}\trow matrix shape: {row_matrix.shape}\tcol matrix shape: {col_matrix.shape}')
                # logger.debug(f'input features: {data}')
                num_vertices = num_vertices[0]
                # data = data.permute(0,2,1)
                batch_size = data.size(0)
                if args.dgcnn:
                    out = dgcnn(data)  # (batch_size, 256, num_points)
                    out = out[:, :, :num_vertices]  # (batch_size, 256, num_vertices)
                else:
                    out = features[:, :num_vertices, :4].transpose(2, 1) # (batch_size, 3, num_vertices)
                # logger.debug(f'output shape: {out.shape}')
                # logger.debug(f'output: {out}')
                pred_row_matrix = torch.zeros([num_vertices, num_vertices], dtype=torch.int).to(device)
                pred_col_matrix = torch.zeros([num_vertices, num_vertices], dtype=torch.int).to(device)
                possible_idx = torch.arange(num_vertices)
                # concatenate tensor row by column

                # combinations = torch.combinations(possible_idx, r=2)  # (num_vertices, num_vertices)
                possible_idx = range(num_vertices)
                combinations = torch.LongTensor(list(itertools.product(possible_idx, possible_idx)))
                row_features1, row_features2 = out[:, :, combinations[:, 0]], out[:, :, combinations[:,
                                                                                        1]]  # (batch_size, 256, combination(num_vertices,2))
                col_features1, col_features2 = out[:, :, combinations[:, 0]], out[:, :, combinations[:,
                                                                                        1]]  # (batch_size, 256, combination(num_vertices, 2))
                row_features1, row_features2, col_features1, col_features2 = row_features1.to(device), row_features2.to(
                    device), col_features1.to(device), col_features2.to(device)
                logger.debug(
                    f'num vertices :{num_vertices}\trow1 features shape: {row_features1.shape}\tcol1 features shape: {col_features1.shape}')
                if args.dgcnn:
                    row_pairs, col_pairs = torch.cat((row_features1, row_features2), 1), torch.cat((col_features1,
                                                                                                    col_features2),
                                                                                                   1)  # (batch_size, 512, combination(num_vertices, 2))
                    logger.debug(f'row pairs shape:  {row_pairs.shape}\tcol pairs shape: {col_pairs.shape}')
                    row_pairs, col_pairs = row_pairs.permute(0, 2, 1), col_pairs.permute(0, 2,
                                                                                         1)  # (batch_size, combination(num_vertices,2), 512)
                else:
                    row_pairs, col_pairs = (row_features1 - row_features2).permute(0, 2, 1), (col_features1 - col_features2).permute(0, 2, 1)
                row_output, col_output = row_classifier(row_pairs), col_classifier(
                    col_pairs)  # (batch_size, combination(num_vertices, 2), 2)
                # logger.debug(f'row output shape: {row_output.shape}')
                # logger.debug(f'row output: {row_output}')
                # row_output, col_output = row_output.permute(0, 2, 1), col_output.permute(0, 2, 1) # (batch_size, combination(num_vertices,2), 2)

                row_idx, col_idx = torch.argmax(row_output, 2), torch.argmax(col_output,
                                                                             2)  # (batch_size, combination(num_vertices, 2))
                logger.debug(f'row_idx shape: {row_idx.shape}')
                non_zero_row, non_zero_col = torch.nonzero(row_idx), torch.nonzero(col_idx)  # (non_zero_num, 2)
                logger.debug(f'non_zero_row shape: {non_zero_row.shape}')
                r_row_idx, c_row_idx, r_col_idx, c_col_idx = non_zero_row[:, 1] // num_vertices, non_zero_row[:,
                                                                                                 1] % num_vertices, non_zero_col[
                                                                                                                    :,
                                                                                                                    1] // num_vertices, non_zero_col[
                                                                                                                                        :,
                                                                                                                                        1] % num_vertices
                logger.debug(f'pred_row_matrix shape: {pred_row_matrix.shape}\tr_row_id shape: {r_row_idx.shape}')
                pred_row_matrix[r_row_idx, c_row_idx] = 1
                pred_col_matrix[r_col_idx, c_col_idx] = 1

                pred_row_matrix = error_correction(pred_row_matrix, row_output)
                pred_col_matrix = error_correction(pred_col_matrix, col_output)
                # compare with ground truth row matrix and column matrix
                gt_row_matrix, gt_col_matrix = row_matrix[0, :num_vertices, :num_vertices], col_matrix[0, :num_vertices,
                                                                                            :num_vertices]
                if args.saved_predicted_dir:
                    chunk_path = '/home/lihuichao/academic/SciTSR/dataset/test/chunk/' + file_path[0]
                    with open(chunk_path, 'r') as f:
                        obj = json.load(f)
                    chunks = obj['chunks']
                    row_fig = draw_rectangle(pred_row_matrix, gt_row_matrix, chunks)
                    col_fig = draw_rectangle(pred_col_matrix, gt_col_matrix, chunks)
                    row_fig.savefig(f'{args.saved_predicted_dir}/{file_path[0]}_row.png')
                    col_fig.savefig(f'{args.saved_predicted_dir}/{file_path[0]}_col.png')
                # pred_row_matrix, pred_col_matrix, gt_row_matrix, gt_col_matrix = pred_row_matrix.cpu(), pred_col_matrix.cpu(), gt_row_matrix.cpu(), gt_col_matrix.cpu()
                # unpack value according to util.metric TP, FP, precision, btp, bfp, bfn, bprecision, brecall
                num_vertices_list.append(num_vertices.item())
                r1, r2, r3, r4, r5, r6, r7, r8 = metric(pred_row_matrix, gt_row_matrix)
                c1, c2, c3, c4, c5, c6, c7, c8 = metric(pred_col_matrix, gt_col_matrix)
                row_TP += r1
                row_FP += r2
                b_row_TP += r4
                b_row_FP += r5
                b_row_FN += r6
                row_macro_precision.append(r3)
                b_row_macro_precision.append(r7)
                b_row_macro_recall.append(r8)

                col_TP += c1
                col_FP += c2
                b_col_TP += c4
                b_col_FP += c5
                b_col_FN += c6
                col_macro_precision.append(c3)
                b_col_macro_precision.append(c7)
                b_col_macro_recall.append(c8)
                logger.info(f'row precision: {r3}\t[b] precision: {r7}\t[b] recall: {r8}')
                logger.info(f'col precision: {c3}\t[b] precision: {c7}\t[b] recall: {c8}')
                # if writer:
                #     writer.add_scalar('Eval/[rb]precision_#vertices', r7, num_vertices)
                #     writer.add_scalar('Eval/[rb]recall_#vertices', r8, num_vertices)
                #     writer.add_scalar('Eval/[cb]precision_#vertices', c7, num_vertices)
                #     writer.add_scalar('Eval/[cb]recall_#vertices', c8, num_vertices)

                # logger.info(f'tp: {row_TP}\tfp: {row_FP}')
                count += 1
        row_macro_precision, b_row_macro_precision, b_row_macro_recall = np.array(row_macro_precision), np.array(b_row_macro_precision), np.array(b_row_macro_recall)
        col_macro_precision, b_col_macro_precision, b_col_macro_recall = np.array(col_macro_precision), np.array(b_col_macro_precision), np.array(b_col_macro_recall)
        num_vertices_list = np.array(num_vertices_list)

        logger.success(
            f'[row]: macro precision: {np.average(row_macro_precision)}\tbinary macro precision: {np.average(b_row_macro_precision)}\tbinary macro recall: {np.average(b_row_macro_recall)}')
        logger.success(
            f'[row]: micro precision: {row_TP / (row_TP + row_FP)}\tbinary micro precision: {b_row_TP / (b_row_TP + b_row_FP)}\tbinary micro recall: {b_row_TP / (b_row_TP + b_row_FN)}')
        logger.success(
            f'[col]: macro precision: {np.average(col_macro_precision)}\tbinary macro precision: {np.average(b_col_macro_precision)}\tmacro recall: {np.average(b_col_macro_recall)}')
        logger.success(
            f'[col]: micro precision: {col_TP / (col_TP + col_FP)}\tbinary micro precision: {b_col_TP / (b_col_TP + b_col_FP)}\tmicro recall: {b_col_TP / (b_col_TP + b_col_FN)}')
        # logger.success(f'average row precision: {row_prec}\taverage columm precision: {col_prec}')
        figure = draw_figure(num_vertices_list, b_row_macro_precision, b_row_macro_recall, b_col_macro_precision,
                             b_col_macro_recall)
        figure.savefig('result_final')
        logger.info('figure saved!')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='scitsr', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--save_step', type=int, default=3000, metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--model_path', type=str,
                        default='/home/lihuichao/academic/dgcnn.pytorch/checkpoints_knn_balanced_normalized/checkpoints_82965',
                        metavar='N',
                        help='The saved model path')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--saved_model_dir', type=str, default='', metavar='N',
                        help='saved model dir')
    parser.add_argument('--log_level', type=str, default='DEBUG', metavar='N', help='level of log, default debug')
    parser.add_argument('--pretrained_model', type=str, default='', metavar='N', help='the pretrained model path')
    parser.add_argument('--summary',type=str, default='', metavar='N', help='tensor board summary directory')
    parser.add_argument('--train_partition',type=str,default='train', metavar='N', help='partition of dataset')
    parser.add_argument('--log', type=str, default='', metavar='N', help='logger file name')
    parser.add_argument('--saved_predicted_dir', type=str, default='', metavar='N', help='dir path of predicted result')
    parser.add_argument('--chunk', type=str, default='chunk_test', metavar='N', help='chunk dir name')
    parser.add_argument('--rel', type=str, default='rel_test', metavar='N', help='rel name')
    parser.add_argument('--dgcnn', action='store_true', default=False)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
