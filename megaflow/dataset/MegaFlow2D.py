import os
import sys
import multiprocessing as mp
from itertools import repeat

import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
import numpy as np
from tqdm import tqdm
from megaflow.common.utils import process_file_list


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
        self.data_list = self.processed_file_names
        if not self.is_processed:
            self.process()

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
        path = download_url(url, dir)
        extract_zip(path, self.root)

    def process(self):
        # Read mesh solution into graph structure
        os.makedirs(self.processed_data_dir, exist_ok=True)
        las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        # split the list according to the number of processors and process the data in parallel
        num_proc = mp.cpu_count()
        las_data_list = np.array_split(las_data_list, num_proc)
        has_data_list = np.array_split(has_data_list, num_proc)
        
        # create a pool of processes
        pool = mp.Pool(num_proc)

        # start the processes
        for i in range(num_proc):
            pool.apply_async(process_file_list, args=(self.raw_data_dir, self.processed_data_dir, las_data_list[i], has_data_list[i]))
        
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

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


class MegaFlow2DSubset(MegaFlow2D):
    """
    This subset splits the entire dataset into 40 subsets, which is initialized via indices.
    """
    def __init__(self, root, indices, transform=None):
        raise NotImplementedError
