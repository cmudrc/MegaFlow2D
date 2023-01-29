import os
import sys
from multiprocessing import Pool, Process

import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
import numpy as np
from tqdm import tqdm


class MegaFlow2D(Dataset):
    """
    The MegaFlow2D dataset is a collection of 2D flow simulations of different geometries.
    Current supported geometries include: circle, ellipse, nozzle.

    Input:
        root: root directory of the dataset
        transform: transform to be applied on the data
        pre_transform: transform to be applied on the data during preprocessing, e.g. splitting into individual graphs 
                    or dividing in temporal sequence
        split_scheme: 'full', 'circle', 'ellipse', 'mixed'
        split_ratio: defult set as [0.5, 0.5] for circle and ellipse respectively
    """
    def __init__(self, root, download, transform=None, pre_transform=None, split_scheme='mixed', split_ratio=None):
        self._indices = None
        self.root = root
        # self.split = split
        self.transforms = transform
        self.pre_transform = pre_transform
        if download:
            self.download()
        self.raw_data_dir = os.path.join(self.root, 'raw')
        self.processed_data_dir = os.path.join(self.root, 'processed')
        if not self.is_processed:
            p = Process(target=self.process)
            p.start()
            p.join()
        self.data_list = self.processed_file_names

        self.circle_data_list = [name for name in self.data_list if name.split('_')[0] == 'circle']
        self.ellipse_data_list = [name for name in self.data_list if name.split('_')[0] == 'ellipse']

        # self.las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        # self.has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # self.mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        self.split_scheme = split_scheme
        if self.split_scheme == 'full':
            self.data_list = self.processed_file_names
        elif self.split_scheme == 'circle':
            self.data_list = self.circle_data_list
        elif self.split_scheme == 'ellipse':
            self.data_list = self.ellipse_data_list
        elif self.split_scheme == 'mixed':
            # split the dataset according to the split_ratio
            if split_ratio is None:
                split_ratio = [0.5, 0.5]
            self.data_list = self.circle_data_list[:int(len(self.circle_data_list) * split_ratio[0])] + \
                                self.ellipse_data_list[:int(len(self.ellipse_data_list) * split_ratio[1])]

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        if os.path.exists(self.processed_data_dir):
            return os.listdir(self.processed_data_dir)
        else:
            return []

    @property
    def is_processed(self):
        return os.path.exists(self.processed_data_dir)

    def len(self):
        return len(self.data_list)

    def download(self, dir=None):
        url = ''
        path = download_url(url, self.root)
        extract_zip(path, self.root)

    def process(self):
        # Read mesh solution into graph structure
        os.makedirs(self.processed_data_dir, exist_ok=True)
        las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        for las_data_name, has_data_name in tqdm(zip(las_data_list, has_data_list)):
            las_data = np.load(os.path.join(self.raw_dir, 'las', las_data_name))
            has_data = np.load(os.path.join(self.raw_dir, 'has', has_data_name))

            str1, str2, str3, str4 = las_data_name.split('_')
            mesh_name = str1 + '_' + str2 + '.npz'
            mesh_data = np.load(os.path.join(self.raw_dir, 'mesh', mesh_name))
            node_data = np.zeros(3)
            val_data = np.zeros(3)
            for j in range(len(mesh_data['x'])):
                node_data[0] = las_data['ux'][j]
                node_data[1] = las_data['uy'][j]
                node_data[2] = las_data['p'][j]

                val_data[0] = has_data['ux'][j]
                val_data[1] = has_data['uy'][j]
                val_data[2] = has_data['p'][j]

                if j == 0:
                    node_data_list = np.array([node_data])
                    val_data_list = np.array([val_data])
                else:
                    node_data_list = np.append(node_data_list, np.array([node_data]), axis=0)
                    val_data_list = np.append(val_data_list, np.array([val_data]), axis=0)

            node_data_list = torch.tensor(node_data_list, dtype=torch.float)
            val_data_list = torch.tensor(val_data_list, dtype=torch.float)
            edge_index = np.array(mesh_data['edges'])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = np.array(mesh_data['edge_properties'])
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            node_pos = np.zeros(2)
            for j in range(len(mesh_data['x'])):
                node_pos[0] = mesh_data['x'][j]
                node_pos[1] = mesh_data['y'][j]

                if j == 0:
                    node_pos_list = np.array([node_pos])
                else:
                    node_pos_list = np.append(node_pos_list, np.array([node_pos]), axis=0)

            node_pos_list = torch.tensor(node_pos_list, dtype=torch.float)

            data = Data(x=node_data_list, y=val_data_list, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, pos=node_pos_list)
            data_name = str1 + '_' + str2 + '_' + str4
            torch.save(data, os.path.join(self.processed_data_dir, data_name + '.pt'))
        self.data_list = self.processed_file_names

    def transform(self, data):
        if self.transforms == 'error_estimation':
            data.y = data.y - data.x
        if self.transforms == 'normalize':
            # normalize the data layer-wise via gaussian distribution
            data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
            data.y = (data.y - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
        return data

    def get(self, idx):
        if not self.is_processed:
            p = Process(target=self.process)
            p.start()
            p.join()
        
        data = torch.load(os.path.join(self.processed_data_dir, self.data_list[idx]))
        
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_eval(self, idx):
        # same as get, but returns data name as well
        if not self.is_processed:
            p = Process(target=self.process)
            p.start()
            p.join()

        data = torch.load(os.path.join(self.processed_data_dir, self.data_list[idx]))
        str1, str2, str4 = self.data_list[idx].split('_')
        data_name = str1 + '_' + str2 + '_' + str4

        if self.transform is not None:
            data = self.transform(data)
        return data, data_name


class MegaFlow2DSubset(MegaFlow2D):
    """
    This subset splits the entire dataset into 40 subsets, which is initialized via indices.
    """
    def __init__(self, root, indices, transform=None):
        raise NotImplementedError
