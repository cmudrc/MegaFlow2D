import os
import sys

import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
import numpy as np

class MegaFlow2D(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, mode='mixed'):
        """
        Initialize the dataset
        :param root: root directory of the dataset
        :param transform: transform to be applied on the data
        :param pre_transform: transform to be applied on the data before saving it to disk
        :param mode: 'mixed' or 'circle' or 'ellipse'
        """
        self._indices = None
        self.root = root
        # self.split = split
        self.transforms = transform
        self.pre_transform = pre_transform
        self.raw_data_dir = os.path.join(self.root, 'raw')
        self.processed_data_dir = os.path.join(self.root, 'processed')
        self.data_list = self.processed_file_names

        self.circle_data_list = [name for name in self.data_list if name.split('_')[0] == 'circle']
        self.ellipse_data_list = [name for name in self.data_list if name.split('_')[0] == 'ellipse']

        # self.las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        # self.has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # self.mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        self.mode = mode
        if self.mode == 'mixed':
            self.data_list = self.processed_file_names
        elif self.mode == 'circle':
            self.data_list = self.circle_data_list
        elif self.mode == 'ellipse':
            self.data_list = self.ellipse_data_list

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

    @property
    def split(self, idx):
        """
        Split the entire dataset into 40 subsets with uniform distribution
        .param idx: index of the subset
        """
        if self._indices is None:
            self._indices = torch.randperm(len(self))
        return self._indices[idx]

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
        for las_data_name, has_data_name in zip(las_data_list, has_data_list):
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
        self.data_list = os.listdir(self.processed_data_dir)

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
            self.process()
        
        data = torch.load(os.path.join(self.processed_data_dir, self.data_list[idx]))
        
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_eval(self, idx):
        # same as get, but returns data name as well
        if not self.is_processed:
            self.process()

        data = torch.load(os.path.join(self.processed_data_dir, self.data_list[idx]))
        str1, str2, str4 = self.data_list[idx].split('_')
        data_name = str1 + '_' + str2 + '_' + str4

        if self.transform is not None:
            data = self.transform(data)
        return data, data_name

