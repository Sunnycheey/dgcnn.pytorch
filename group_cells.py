# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 4:41 下午
# @Author  : lihuichao
# @File    : group_cells.py
# @Project: JSP

import numpy as np
import json
import networkx as nx
import torch
import os
import shutil
from tqdm import tqdm
from loguru import logger
from util import draw_rectangle


class Cell:
    def __init__(self, x1, x2, y1, y2, text):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.x_center = x1 + (x2 - x1) / 2
        self.y_center = y1 + (y2 - y1) / 2
        self.text = text
        self.width = x2 - x1
        self.height = y2 - y1

    def x_abs_distance(self, cell):
        x1_distance = abs(cell.x1 - self.x1)
        x2_distance = abs(cell.x2 - self.x2)
        return x1_distance, x2_distance

    def y_abs_distance(self, cell):
        y1_distance = abs(cell.y1 - self.y1)
        y2_distance = abs(cell.y2 - self.y2)
        return y1_distance, y2_distance

    def center_distance(self, cell):
        x_center_distance = abs(cell.x_center - self.x_center)
        y_center_distance = abs(cell.y_center - self.y_center)
        return x_center_distance, y_center_distance


class Cells:
    def __init__(self, chunk_file_path, x_epsilon, y_epsilon, center_epsilon):
        # Todo: set suitable x_epsilon and y_epsilon
        with open(chunk_file_path, 'r') as f:
            obj = json.load(f)
        chunks = obj['chunks']
        cells = []
        cell_feature = np.empty([len(chunks), 4])
        for i, chunk in enumerate(chunks):
            pos = chunk['pos']
            c = Cell(pos[0], pos[1], pos[2], pos[3], chunk['text'])
            cell_feature[i] = pos[0], pos[1], pos[2], pos[3]
            cells.append(c)
        self.cells = cells
        self.x_epsilon = x_epsilon
        self.y_epsilon = y_epsilon
        self.center_epsilon = center_epsilon
        self.row_group, self.column_group = self.group_cells()
        self.cells_feature = cell_feature  # (len(chunks), 4)
        self.chunks = chunks

    def group_cells(self):
        """
        group cells by its x, y coordinate :param cells: a list of cells (chunk) :param x_epsilon: epsilon in x
        direction :param y_epsilon: epsilon in y direction :return: the two grouped cells index list (x direction and
        y direction), ordered by x coordinate then y coordinate
        """
        x_group_lists, y_group_lists = [], []
        n_cells = len(self.cells)
        xg, yg = nx.Graph(), nx.Graph()
        xg.add_nodes_from(range(n_cells))
        yg.add_nodes_from(range(n_cells))
        x1_dis, x2_dis = np.empty([n_cells, n_cells]), np.empty([n_cells, n_cells])
        y1_dis, y2_dis = np.empty([n_cells, n_cells]), np.empty([n_cells, n_cells])
        x_center_dis, y_center_dis = np.empty([n_cells, n_cells]), np.empty(([n_cells, n_cells]))
        width, height = np.empty([n_cells]), np.empty([n_cells])
        x_used, y_used = np.zeros([n_cells], dtype=np.bool), np.zeros([n_cells], dtype=np.bool)
        for i, cell1 in enumerate(self.cells):
            for j, cell2 in enumerate(self.cells):
                x1_dis[i][j], x2_dis[i][j] = cell1.x_abs_distance(cell2)
                y1_dis[i][j], y2_dis[i][j] = cell1.y_abs_distance(cell2)
                x_center_dis[i][j], y_center_dis[i][j] = cell1.center_distance(cell2)
                width[i] = cell1.width
                height[i] = cell1.height
                # 包含关系一定是同组关系
                if cell1.x1 <= cell2.x1 and cell1.x2 >= cell2.x2:
                    xg.add_edge(i, j)
                if cell1.y1 <= cell2.y1 and cell1.y2 >= cell2.y2:
                    yg.add_edge(i, j)
        x1_idx, x2_idx = x1_dis < self.x_epsilon, x2_dis < self.x_epsilon
        y1_idx, y2_idx = y1_dis < self.y_epsilon, y2_dis < self.y_epsilon
        x_center_idx, y_center_idx = x_center_dis < self.center_epsilon, y_center_dis < self.center_epsilon
        x_idx, y_idx = np.logical_or(x1_idx, x2_idx, x_center_idx), np.logical_or(y1_idx, y2_idx, y_center_idx)
        # utilizing graph to group the connected relation
        # Todo: 解决联通性导致同行的cell在列矩阵中关联 （貌似只需要在预测的时候处理）
        x_edges, y_edges = zip(np.nonzero(x_idx)[0], np.nonzero(x_idx)[1]), zip(np.nonzero(y_idx)[0],
                                                                                np.nonzero(y_idx)[1])
        xg.add_edges_from(x_edges)
        yg.add_edges_from(y_edges)
        width_sort_idx, height_sort_idx = np.argsort(width), np.argsort(height)
        for idx in width_sort_idx:
            if not x_used[idx]:
                neighbors = [int(n) for n in xg.neighbors(idx)]
                x_used[neighbors] = True
                x_group_lists.append(neighbors)
        for idx in height_sort_idx:
            if not y_used[idx]:
                neighbors = [int(n) for n in yg.neighbors(idx)]
                y_used[neighbors] = True
                y_group_lists.append(neighbors)
        return y_group_lists, x_group_lists

        # for s in nx.connected_components(xg):
        #     x_group_lists.append(s)
        # for s in nx.connected_components(yg):
        #     y_group_lists.append(s)
        # return y_group_lists, x_group_lists

    def group_feature_extractor(self, group_idx_list, group_type):
        """
        extract group feature for group
        :param group_idx_list: the group index list in cells
        :param group_type: 0 for row group, 1 for column group
        :return: the fixed size feature of certain group
        """
        all_features = self.cells_feature[group_idx_list]
        if group_type == 0:
            start_x = np.min(all_features[:, 0])
            end_x = np.max(all_features[:, 1])
            avg_y1 = np.average(all_features[:, 2])
            avg_y2 = np.average(all_features[:, 3])
            return np.array([avg_y1, avg_y2, start_x, end_x])
        if group_type == 1:
            start_y = np.min(all_features[:, 2], 0)
            end_y = np.max(all_features[:, 3])
            avg_x1 = np.average(all_features[:, 0])
            avg_x2 = np.average(all_features[:, 1])
            return np.array([avg_x1, avg_x2, start_y, end_y])

    def convert_to_torch_matrix(self, group_lists, device='cpu'):
        """
        convert the grouped cells to matrix representation (torch.tensor)
        :param device: torch tensor device
        :return: matrix
        """
        matrix = torch.zeros([len(self.cells), len(self.cells)], device=device, dtype=torch.int)
        for group_list in group_lists:
            group_list = list(group_list)
            for i in range(len(group_list) - 1):
                matrix[group_list[i]][group_list[i + 1]] = 1
                matrix[group_list[i + 1]][group_list[i]] = 1
        return matrix

    def draw_matrix(self, device='cpu'):
        gt = torch.zeros([len(self.cells), len(self.cells)], device=device, dtype=torch.int)
        row_matrix = self.convert_to_torch_matrix(self.row_group)
        col_matrix = self.convert_to_torch_matrix(self.column_group)
        row_fig = draw_rectangle(row_matrix, gt, self.chunks)
        col_fig = draw_rectangle(col_matrix, gt, self.chunks)
        return row_fig, col_fig

    def sort_group_by_coords(self, group, direction):
        """
        sort group by coordinate
        :param group: group index
        :param direction: the sort direction, x or y (horizontally or vertically)
        :return:
        """
        pass


class Dataset:
    # Todo: 对数据集进行预处理，使其满足定义，先找出没有划分出组的节点进行匹配
    def __init__(self, chunks_dir, saved_dir, dataset_dir, partition):
        self.chunks_dir = chunks_dir
        self.saved_dir = saved_dir
        self.dataset_dir = dataset_dir
        self.partition = partition
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        else:
            shutil.rmtree(saved_dir)
            os.makedirs(saved_dir)

    def iterate_data(self):
        for file_path in os.listdir(self.chunks_dir):
            full_path = os.path.join(self.chunks_dir, file_path)

        for file_path in os.listdir(self.chunks_dir):
            full_path = os.path.join(self.chunks_dir, file_path)
            c = Cells(full_path, 1.5, 1.5, 1.5)
            row_fig, col_fig = c.draw_matrix()
            row_fig.savefig(f'{os.path.join(self.saved_dir, file_path)}_row.png')
            col_fig.savefig(f'{os.path.join(self.saved_dir, file_path)}_col.png')
    @logger.catch
    def preprocess(self):
        main_folder = os.path.join(self.dataset_dir, self.partition)
        chunk_dir = os.path.join(main_folder, 'chunk')
        rel_dir = os.path.join(main_folder, 'rel')
        logger.info(f'Start loading data...')
        for chunk_file_path in tqdm(os.listdir(chunk_dir)):
            if chunk_file_path.endswith('.ipynb_checkpoints'):
                continue
            ret = {'row_rel': {}, 'col_rel': {}}
            chunk_id = chunk_file_path.split('.')[0:-1]
            chunk_id = '.'.join(chunk_id)
            cells = Cells(os.path.join(chunk_dir, chunk_file_path), 1.5, 1.5, 1.5)
            row_group, column_group = cells.row_group, cells.column_group
            row_rel, col_rel = nx.Graph(), nx.Graph()
            row_rel.add_nodes_from(range(len(cells.chunks)))
            col_rel.add_nodes_from(range(len(cells.chunks)))
            with open(os.path.join(rel_dir, f'{chunk_id}.rel')) as f:
                for line in f:
                    start, end, rel_type = line.split('\t')
                    start, end = int(start), int(end)
                    rel_type = rel_type.split(':')[0]
                    if rel_type == '1':
                        row_rel.add_edge(start, end)
                    if rel_type == '2':
                        col_rel.add_edge(start, end)
            for s in row_group:
                if len(s) == 1:
                    logger.info(f'single element in row')
                    # unmatched node
                    neighbors = row_rel.neighbors(s[0])
                    for neighbor in neighbors:
                        for rel in row_group:
                            if neighbor in rel:
                                if not s[0] in ret['row_rel'].keys():
                                    ret['row_rel'][s[0]] = []
                                ret['row_rel'][s[0]].append(rel)

            for s in column_group:
                if len(s) == 1:
                    logger.info(f'single element in column')
                    # unmatched node
                    neighbors = col_rel.neighbors(s[0])
                    for neighbor in neighbors:
                        for rel in column_group:
                            if neighbor in rel:
                                if not s[0] in ret['col_rel'].keys():
                                    ret['col_rel'][s[0]] = []
                                ret['col_rel'][s[0]].append(rel)

            if len(ret['row_rel']) > 0 or len(ret['col_rel']) > 0:
                with open(f'{os.path.join(self.saved_dir, chunk_id)}.rel', 'w') as f:
                    json.dump(ret, f)

    def dict_post_process(self, dic):
        for key in dic.keys():
            v_list = dic[key]




if __name__ == '__main__':
    # cs = Cells('/home/lihuichao/academic/SciTSR/dataset/test/chunk/0001020v1.12.chunk', 1.5, 1.5)
    # print(cs.group_cells())
    d = Dataset('/home/lihuichao/academic/SciTSR/dataset/test/chunk',
                '/home/lihuichao/academic/SciTSR/dataset/test/new_rel', '/home/lihuichao/academic/SciTSR/dataset', 'test')
    d.preprocess()
