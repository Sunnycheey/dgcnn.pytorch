#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/2/27 9:32 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
import torch
import networkx as nx
import itertools
from torch.utils.data import Dataset
from tqdm import tqdm
from loguru import logger


def get_all_paths_from_graph(G: nx.DiGraph):
    ret = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target and G.in_degree(source) == 0 and G.out_degree(target) == 0:
                try:
                    paths = nx.all_simple_paths(G, source, target)
                    paths = list(paths)
                    if len(paths) >= 2:
                        logger.warning(f'exists {len(paths)} path between source and target')
                    if len(paths) >= 10000:
                        logger.warning(f'too many path: {len(paths)}')
                        return []
                    ret.extend(list(paths))
                except nx.NetworkXNoPath as e:
                    pass
    return ret


def pre_process(rel_dir: str, output_dir: str):
    """
    preprocess the SciTSR dataset
    :param rel_dir relation directory
    :param output_dir the result dir
    :return: None
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_path in tqdm(os.listdir(rel_dir)):
        full_path = os.path.join(rel_dir, file_path)
        logger.info(f'file path: {full_path}')
        row_G = nx.DiGraph()
        col_G = nx.DiGraph()
        lines = []
        row_rels, col_rels = [], []
        with open(full_path, 'r') as f:
            for line in f:
                start, end, rel_type = line.split('\t')
                rel_type = rel_type.split(':')[0]
                start, end = int(start), int(end)
                if rel_type == '1':
                    row_G.add_node(start)
                    row_G.add_node(end)
                    row_G.add_edge(start, end)
                elif rel_type == '2':
                    col_G.add_node(start)
                    col_G.add_node(end)
                    col_G.add_edge(start, end)
            # if not graph_assert(row_G): logger.info(file_path)
            # if not graph_assert(col_G): logger.info(file_path)
        for row_points_set in get_all_paths_from_graph(row_G):
            row_pairs = itertools.permutations(row_points_set, 2)
            for (v1, v2) in row_pairs:
                lines.append(f'{v1}\t{v2}\t1:0')
        for col_points_set in get_all_paths_from_graph(col_G):
            col_pairs = itertools.permutations(col_points_set, 2)
            for (v1, v2) in col_pairs:
                lines.append(f'{v1}\t{v2}\t2:0')
        with open(f'{output_dir}/{file_path}', 'w') as f:
            f.write('\n'.join([line for line in lines]))


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('rm %s' % (zippath))


def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5' % partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


def load_scitsr_data(dataset_dir, partition, padding=False, max_vertice_num=512):
    """
    load scitsr dataset
    :param dataset_dir: dataset dir path
    :param partition: partition of dataset: train or test
    :param padding: whether padding the dataset with max_vertice_num
    :param max_vertice_num: the padding size for dataset
    :param feature_size: input feature size
    :return: the loaded dataset
    """
    import json

    data, row_matrices, col_matrices = [], [], []
    CHUNK = 'chunk'
    REL = 'rel'
    main_folder = os.path.join(dataset_dir, partition)
    chunk_dir = os.path.join(main_folder, CHUNK)
    rel_dir = os.path.join(main_folder, REL)
    logger.info(f'Start loading data...')
    for chunk_file_path in tqdm(os.listdir(chunk_dir)):
        if chunk_file_path.endswith('.ipynb_checkpoints'):
            continue
        chunk_id = chunk_file_path.split('.')[0:-1]
        chunk_id = '.'.join(chunk_id)
        with open(os.path.join(chunk_dir, chunk_file_path)) as f:
            obj = json.load(f)
            chunks = obj['chunks']
            num_points = len(chunks)
            coords = np.zeros([max_vertice_num, 4]) if padding else np.zeros([num_points, 4])
            chunk_text = []
            if num_points > max_vertice_num and padding:
                continue
            i = 0
            for chunk in chunks:
                coords[i] = chunk['pos'][0], chunk['pos'][1], chunk['pos'][2], chunk['pos'][3]
                chunk_text.append(chunk['text'])
                i += 1
        if padding:
            row_matrix, col_matrix = torch.zeros([max_vertice_num, max_vertice_num], dtype=torch.int), torch.zeros(
                [max_vertice_num, max_vertice_num], dtype=torch.int)
        else:
            row_matrix, col_matrix = torch.zeros([num_points, num_points], dtype=torch.int), torch.zeros(
                [num_points, num_points], dtype=torch.int)
        with open(os.path.join(rel_dir, f'{chunk_id}.rel')) as f:
            for line in f:
                start, end, rel_type = line.split('\t')
                start, end = int(start), int(end)
                rel_type = rel_type.split(':')[0]
                if rel_type == '1':
                    row_matrix[start, end] = 1
                    row_matrix[end, start] = 1
                elif rel_type == '2':
                    col_matrix[start, end] = 1
                    col_matrix[end, start] = 1
        # if padding:
        #     data.append({'features': np.array(feature), 'row_matrix': row_matrix, 'col_matrix': col_matrix, 'num_vertices': num_points})
        # else:
        coords = torch.from_numpy(coords)
        tab_w, tab_h = coords[:, 1].max() - coords[:, 0].min(), coords[:, 3].max() - coords[:, 2].min()
        x_center, y_center = (coords[:, 0] + coords[:, 1]) / 2, (coords[:, 2] + coords[:, 3]) / 2
        relative_x1, relative_x2, relative_y1, relative_y2 = coords[:, 0] / tab_w, coords[:, 1] / tab_w, coords[
                                                                                                         :,
                                                                                                         2] / tab_h, coords[
                                                                                                                     :,
                                                                                                                     3] / tab_h
        relative_x_center, relative_y_center = (coords[:, 0] + coords[:, 1]) / 2 / tab_w, (
                coords[:, 2] + coords[:, 3]) / 2 / tab_h
        height_of_chunk, width_of_chunk = coords[:, 3] - coords[:, 2], coords[:, 1] - coords[:, 0]
        # tab_w_list, tab_h_list = np.zeros([])
        x_center, y_center, relative_x1, relative_x2, relative_y1, relative_y2, relative_x_center, relative_y_center, height_of_chunk, width_of_chunk = \
            x_center.unsqueeze(1), y_center.unsqueeze(1), relative_x1.unsqueeze(1), relative_x2.unsqueeze(
                1), relative_y1.unsqueeze(1), relative_y2.unsqueeze(1), relative_x_center.unsqueeze(
                1), relative_y_center.unsqueeze(1), height_of_chunk.unsqueeze(1), width_of_chunk.unsqueeze(1)
        feature = torch.cat((coords, x_center, y_center, relative_x1, relative_x2, relative_y1,
                             relative_y2, relative_x_center, relative_y_center, height_of_chunk, width_of_chunk),
                            dim=1)
        data.append({'file_path': chunk_file_path, 'features': feature, 'row_matrix': row_matrix,
                     'col_matrix': col_matrix, 'num_vertices': num_points})

    return data


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class SciTSR(Dataset):
    def __init__(self, dataset_dir, max_vertice_num=1024, feature_size=15, partition='train', normalize=False,
                 mean=None, std=None):
        self.data = load_scitsr_data(dataset_dir, partition,
                                     max_vertice_num=max_vertice_num)  # data: [num_points * feature_size], row/col_matix: [num_points, num_points]
        self.feature_size = feature_size
        self.max_vertice_num = max_vertice_num
        self.partition = partition
        self.normalize = normalize
        if normalize:
            self.mean = torch.tensor(
                [316.78361649456815, 346.4171709500245, 456.7999285136674, 461.4904624557267, 331.60039372229636,
                 459.145195484697, 1.0708147421800533, 1.172230501966359, 4.613889995626876, 4.661217907288215,
                 1.1215226220731356, 4.637553951457586, 4.690533942059325, 29.633554455456355])
            self.std = torch.tensor(
                [111.50166485622228, 109.23567514593609, 87.45144478458685, 87.45525629652187, 109.27551704485022,
                 87.45231689518513, 0.48120236318145115, 0.4996407959940434, 4.125667755990256, 4.164269755774229,
                 0.48757244649731324, 4.144962765395593, 0.8503987020471627, 31.073357274697436])

    def __getitem__(self, i):
        if self.normalize:
            self.data[i]['features'].sub_(self.mean).div_(self.std)
        return self.data[i]

    def __len__(self):
        return len(self.data)


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


class Vertex(object):

    def __init__(self, vid: int, chunk, tab_h, tab_w):
        """
    Args:
      vid: Vertex id
      chunk: the chunk to extract features
      tab_h: height of the table (y-axis)
      tab_w: width of the table (x-axis)
    """
        self.tab_h = tab_h
        self.tab_w = tab_w
        self.chunk = chunk
        self.features = self.get_features()

    def get_features(self):
        return {
            "x1": self.chunk.x1,
            "x2": self.chunk.x2,
            "y1": self.chunk.y1,
            "y2": self.chunk.y2,
            "x center": (self.chunk.x1 + self.chunk.x2) / 2,
            "y center": (self.chunk.y2 + self.chunk.y2) / 2,
            "relative x1": self.chunk.x1 / self.tab_w,
            "relative x2": self.chunk.x2 / self.tab_w,
            "relative y1": self.chunk.y1 / self.tab_h,
            "relative y2": self.chunk.y2 / self.tab_h,
            "relative x center": (self.chunk.x1 + self.chunk.x2) / 2 / self.tab_w,
            "relative y center": (self.chunk.y2 + self.chunk.y2) / 2 / self.tab_h,
            "height of chunk": self.chunk.y2 - self.chunk.y1,
            "width of chunk": self.chunk.x2 - self.chunk.x1
        }


class Chunk:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # data, label = train[0]
    # print(data.shape)
    # print(label.shape)
    #
    # trainval = ShapeNetPart(2048, 'trainval')
    # test = ShapeNetPart(2048, 'test')
    # data, label, seg = trainval[0]
    # print(data.shape)
    # print(label.shape)
    # print(seg.shape)
    #
    # train = S3DIS(4096)
    # test = S3DIS(4096, 'test')
    # data, seg = train[0]
    # print(data.shape)
    # print(seg.shape)

    # records = load_scitsr_data('/home/lihuichao/academic/SciTSR/dataset', 'train')
    # feature = None
    # for i, record in enumerate(records):
    #     if feature is None:
    #         feature = record['features']
    #     else:
    #         feature = torch.cat((feature, record['features']), dim=0)
    # mean, std = torch.mean(feature, dim=0), torch.std(feature, dim=0)
    # std += 1e-6
    # print(mean.tolist())
    # print(std.tolist())
    # feature.sub_(mean).div_(std)
    # logger.info(feature)
    pre_process('/home/lihuichao/academic/SciTSR/dataset/train/rel_original',
                '/home/lihuichao/academic/SciTSR/dataset/train/rel_new_new')
