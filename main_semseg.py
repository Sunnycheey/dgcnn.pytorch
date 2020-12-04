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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DIS, SciTSR
from model import DGCNN_semseg, PairClassfier
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import shutil
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(sys.stdout, level="INFO")


def _init_():
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
    train_loader = DataLoader(SciTSR(partition='train', dataset_dir='/home/lihuichao/academic/SciTSR/dataset'),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(SciTSR(partition='test', dataset_dir='/home/lihuichao/academic/SciTSR/dataset'),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args, 5).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    if args.clear:
        shutil.rmtree('checkpoints')
        os.makedirs('checkpoints')

    row_classifier, col_classifier = PairClassfier(512).to(device), PairClassfier(512).to(device)
    row_classifier, col_classifier = nn.DataParallel(row_classifier), nn.DataParallel(col_classifier)

    model = nn.DataParallel(model)

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
    # best_test_iou = 0
    for epoch in tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        row_classifier.train()
        col_classifier.train()
        for data in tqdm(train_loader):
            iterate_step += 1
            features, row_matrix, col_matrix, num_vertices = data['features'].type(torch.FloatTensor), data[
                'row_matrix'].type(torch.FloatTensor), data['col_matrix'].type(torch.FloatTensor), data['num_vertices']
            data, row_matrix, col_matrix, num_vertices = features.to(device), row_matrix.to(device), col_matrix.to(
                device), num_vertices.to(device)

            # data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            all_features = model(data)  # (batch_size, 256, max_num_vertices)

            non_zero_row = torch.nonzero(row_matrix)
            non_zero_col = torch.nonzero(col_matrix)
            zero_row = torch.nonzero((row_matrix == 0)).to(device)
            zero_col = torch.nonzero((col_matrix == 0)).to(device)
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

            row_neg_point = torch.zeros([row_input_shape, 256]).to(device)
            if row_input_shape > row_sampling.size(0):
                logger.warning(f'row negative sampling padding with zero vector')
            row_neg_point[:row_sampling.size(0)] = row_point3[row_sampling]

            # row_point4 = row_point4[:row_input_shape] if row_point4.size(0) > (row_input_shape) else row_point4
            col_neg_point = torch.zeros([col_input_shape, 256]).to(device)
            if col_input_shape > col_sampling.size(0):
                logger.warning(f'col negative sampling padding with zero vector')
            col_neg_point[:col_sampling.size(0)] = col_point3[col_sampling]
            # col_point4 = col_point4[:col_input_shape] if col_point4.size(0) > (col_input_shape) else col_point4

            # Todo: sampling from row and col

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
            row_input = torch.cat((row_pos_input, row_neg_input), 0)
            col_input = torch.cat((col_pos_input, col_neg_input), 0)

            row_pos_gt, row_neg_gt = torch.ones([row_input_shape], dtype=torch.long), torch.zeros(
                [row_neg_input.size(0)],
                dtype=torch.long)
            col_pos_gt, col_neg_gt = torch.ones([col_input_shape], dtype=torch.long), torch.zeros(
                [col_neg_input.size(0)],
                dtype=torch.long)

            row_pos_gt, row_neg_gt = row_pos_gt.to(device), row_neg_gt.to(device)
            col_pos_gt, col_neg_gt = col_pos_gt.to(device), col_neg_gt.to(device)
            row_gt = torch.cat((row_pos_gt, row_neg_gt), 0)
            col_gt = torch.cat((col_pos_gt, col_neg_gt), 0)

            # print(f'col gt shape: {col_gt.shape}')

            # g = classifier(row_input)
            row_output = F.softmax(row_classifier(row_input), 1)
            col_output = F.softmax(col_classifier(col_input), 1)

            loss = F.cross_entropy(row_output, row_gt) + F.cross_entropy(col_output, col_gt)
            logger.info(f'loss: {loss}')
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
        torch.save({'dgcnn': model.state_dict(), 'row_classifier': row_classifier.state_dict(),
                    'col_classifier': col_classifier.state_dict()},
                   f'{args.saved_model_dir}/checkpoints_{iterate_step}')


        # seg_np = seg.cpu().numpy()  # (batch_size, num_points)
        # pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
        # train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
        # train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
        # train_true_seg.append(seg_np)
        # train_pred_seg.append(pred_np)
        # if args.scheduler == 'cos':
        #     scheduler.step()
        # elif args.scheduler == 'step':
        #     if opt.param_groups[0]['lr'] > 1e-5:
        #         scheduler.step()
        #     if opt.param_groups[0]['lr'] < 1e-5:
        #         for param_group in opt.param_groups:
        #             param_group['lr'] = 1e-5
        # train_true_cls = np.concatenate(train_true_cls)
        # train_pred_cls = np.concatenate(train_pred_cls)
        # train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        # avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        # train_true_seg = np.concatenate(train_true_seg, axis=0)
        # train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        # train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f ' % (epoch, train_loss * 1.0 / count)
        io.cprint(outstr)
        #
        # ####################
        # # Test
        # ####################
        # test_loss = 0.0
        # count = 0.0
        # model.eval()
        # test_true_cls = []
        # test_pred_cls = []
        # test_true_seg = []
        # test_pred_seg = []
        # for data, seg in test_loader:
        #     data, seg = data.to(device), seg.to(device)
        #     data = data.permute(0, 2, 1)
        #     batch_size = data.size()[0]
        #     seg_pred = model(data)
        #     seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        #     loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        #     pred = seg_pred.max(dim=2)[1]
        #     count += batch_size
        #     test_loss += loss.item() * batch_size
        #     seg_np = seg.cpu().numpy()
        #     pred_np = pred.detach().cpu().numpy()
        #     test_true_cls.append(seg_np.reshape(-1))
        #     test_pred_cls.append(pred_np.reshape(-1))
        #     test_true_seg.append(seg_np)
        #     test_pred_seg.append(pred_np)
        # test_true_cls = np.concatenate(test_true_cls)
        # test_pred_cls = np.concatenate(test_pred_cls)
        # test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        # avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        # test_true_seg = np.concatenate(test_true_seg, axis=0)
        # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        # test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        # outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
        #                                                                                       test_loss * 1.0 / count,
        #                                                                                       test_acc,
        #                                                                                       avg_per_class_acc,
        #                                                                                       np.mean(test_ious))
        # io.cprint(outstr)
        # if np.mean(test_ious) >= best_test_iou:
        #     best_test_iou = np.mean(test_ious)
        #     torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, args.test_area))


def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []

    test_loader = DataLoader(SciTSR(partition='test', dataset_dir='/home/lihuichao/academic/SciTSR/dataset'),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():

        device = torch.device("cuda" if args.cuda else "cpu")
        dgcnn = DGCNN_semseg(args, 5).to(device)
        dgcnn = nn.DataParallel(dgcnn)
        model_dict = torch.load(args.model_path)
        dgcnn.load_state_dict(model_dict['dgcnn'])
        row_classifier, col_classifier = PairClassfier(512), PairClassfier(512)
        row_classifier, col_classifier = nn.DataParallel(row_classifier), nn.DataParallel(col_classifier)
        row_classifier.load_state_dict(model_dict['row_classifier'])
        col_classifier.load_state_dict(model_dict['col_classifier'])
        print(row_classifier)
        dgcnn, row_classifier, col_classifier = dgcnn.to(device), row_classifier.to(device), col_classifier.to(device)
        dgcnn, row_classifier, col_classifier = dgcnn.eval(), row_classifier.eval(), col_classifier.eval()
        count = 0
        row_prec, col_prec = 0, 0
        for data in tqdm(test_loader):
            features, row_matrix, col_matrix, num_vertices = data['features'].type(torch.FloatTensor), data[
                'row_matrix'].type(torch.FloatTensor), data['col_matrix'].type(torch.FloatTensor), data['num_vertices']
            data, row_matrix, col_matrix, num_vertices = features.to(device), row_matrix.to(device), col_matrix.to(
                device), num_vertices.to(device)
            logger.info(f'num_vertices shape: {num_vertices.shape}\tdata shape: {data.shape}\trow matrix shape: {row_matrix.shape}\tcol matrix shape: {col_matrix.shape}')
            num_vertices = num_vertices[0]
            # data = data.permute(0,2,1)
            batch_size = data.size(0)
            out = dgcnn(data)  # (batch_size, 256, num_points)
            out = out[:, :, :num_vertices]  # (batch_size, 256, num_vertices)
            pred_row_matrix = torch.zeros([num_vertices, num_vertices], dtype=torch.int).to(device)
            pred_col_matrix = torch.zeros([num_vertices, num_vertices], dtype=torch.int).to(device)
            possible_idx = torch.arange(num_vertices)
            # concatenate tensor row by column
            combinations = torch.combinations(possible_idx, r=2)  # (num_vertices, num_vertices)
            row_features1, row_features2 = out[:, :, combinations[:, 0]], out[:, :, combinations[:,
                                                                                    1]]  # (batch_size, 256, combination(num_vertices,2))
            col_features1, col_features2 = out[:, :, combinations[:, 0]], out[:, :, combinations[:,
                                                                                    1]]  # (batch_size, 256, combination(num_vertices, 2))
            row_features1, row_features2, col_features1, col_features2 = row_features1.to(device), row_features2.to(device), col_features1.to(device), col_features2.to(device)
            logger.debug(f'num vertices :{num_vertices}\trow1 features shape: {row_features1.shape}\tcol1 features shape: {col_features1.shape}')
            row_pairs, col_pairs = torch.cat((row_features1, row_features2), 1), torch.cat((col_features1,
                                                                                      col_features2), 1)  # (batch_size, 512, combination(num_vertices, 2))
            logger.debug(f'row pairs shape:  {row_pairs.shape}\tcol pairs shape: {col_pairs.shape}')
            row_pairs, col_pairs = row_pairs.permute(0,2,1), col_pairs.permute(0,2,1) # (batch_size, combination(num_vertices,2), 512)
            row_output, col_output = row_classifier(row_pairs), col_classifier(
                col_pairs)  # (batch_size, combination(num_vertices, 2), 2)
            # row_output, col_output = row_output.permute(0, 2, 1), col_output.permute(0, 2, 1) # (batch_size, combination(num_vertices,2), 2)

            row_idx, col_idx = torch.argmax(row_output, 2), torch.argmax(col_output, 2) # (batch_size, combination(num_vertices, 2))
            logger.debug(f'row_idx shape: {row_idx.shape}')
            non_zero_row, non_zero_col = torch.nonzero(row_idx), torch.nonzero(col_idx) # (non_zero_num, 2)
            logger.debug(f'non_zero_row shape: {non_zero_row.shape}')
            r_row_idx, c_row_idx, r_col_idx, c_col_idx = non_zero_row[:, 1] // num_vertices, non_zero_row[:,1] % num_vertices, non_zero_col[:,1] // num_vertices, non_zero_col[:,1] % num_vertices
            logger.debug(f'pred_row_matrix shape: {pred_row_matrix.shape}\tr_row_id shape: {r_row_idx.shape}')
            pred_row_matrix[r_row_idx, c_row_idx] = 1
            pred_col_matrix[r_col_idx, c_col_idx] = 1

            # compare with ground truth row matrix and column matrix
            gt_row_matrix, gt_col_matrix = row_matrix[:, :num_vertices, :num_vertices], col_matrix[:, :num_vertices, :num_vertices]

            row_prec = torch.true_divide(torch.sum(torch.eq(gt_row_matrix, pred_row_matrix)), (num_vertices * num_vertices)).item()
            col_prec = torch.true_divide(torch.sum(torch.eq(gt_col_matrix, pred_col_matrix)), (num_vertices * num_vertices)).item()
            logger.success(f'row matrix precision: {row_prec}\t#relation: {torch.sum(pred_row_matrix)}')
            logger.success(f'col matrix precision: {col_prec}\t#relation: {torch.sum(pred_col_matrix)}')
            count += 1
        logger.success(f'average row precision: {row_prec}\taverage columm precision: {col_prec}')


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
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
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
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=512,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--save_step', type=int, default=600, metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--model_path', type=str, default='/home/lihuichao/academic/dgcnn.pytorch/checkpoints/checkpoints_6910', metavar='N',
                        help='The saved model path')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--saved_model_dir', type=str, default='', metavar='N',
                        help='saved model dir')
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
