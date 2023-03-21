import os
import sys
import time
import multiprocessing as mp
from threading import Thread
import h5py
from tqdm import tqdm

import torch
from torch_geometric.data import Data, Dataset, download_url, extract_zip
import numpy as np
from megaflow.common.utils import process_file_list, update_progress, copy_group, merge_hdf5_files


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
    def __init__(self, root, download, transform, pre_transform, split_scheme='mixed', split_ratio=None):
        self._indices = None
        self.root = root
        # self.split = split
        self.transforms = transform
        self.pre_transform = pre_transform
        if download:
            self.download()
        self.raw_data_dir = os.path.join(self.root, 'raw')
        self.data_list = self.get_data_list
        self.processed_data_dir = os.path.join(self.root, 'processed')
        # self.processed_las_data_dir = os.path.join(self.root, 'processed', 'las')
        # self.processed_has_data_dir = os.path.join(self.root, 'processed', 'has')
        
        if not self.is_processed:
            self.process()
        # input_file = [os.path.join(self.processed_data_dir, 'data_{}.h5'.format(i)) for i in range(24)]
        # merge_hdf5_files(input_files=input_file, output_file=os.path.join(self.processed_data_dir, 'data.h5'))

        self.circle_data_list = [name for name in self.data_list if name.split('_')[0] == 'circle']
        self.ellipse_data_list = [name for name in self.data_list if name.split('_')[0] == 'ellipse']

        # self.circle_low_res_data_list = [name for name in self.data_list if name.split('_')[0] == 'las']
        # self.high_res_data_list = [name for name in self.data_list if name.split('_')[0] == 'has']

        # self.las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        # self.has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # self.mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        self.split_scheme = split_scheme
        if self.split_scheme == 'full':
            self.data_list = self.data_list
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
        return os.listdir(os.path.join(self.raw_data_dir, 'las'))

    @property
    def processed_file_names(self):
        if os.path.exists(self.processed_data_dir):
            return os.listdir(self.processed_data_dir)
        else:
            return []

    @property
    def is_processed(self):
        if os.path.exists(self.processed_data_dir):
            if len(self.processed_file_names) == 0:
                return False
            else:
                return True
        else:
            return False
        
    @property
    def get_data_list(self):
        # process raw file names into geometry_index_timestep format, save the list in data_list
        raw_file_names = self.raw_file_names
        _data_list = []
        for file_name in raw_file_names:
            str1, str2, str3, str4 = file_name.split('_')
            str4 = str4.split('.')[0]
            _data_list.append(str1 + '_' + str2 + '_' + str4)

        return _data_list
    
    def len(self):
        if not self.is_processed:
            return 0
        else:
            return len(self.data_list)

    def download(self):
        url = 'https://huggingface.co/datasets/cmudrc/MegaFlow2D/resolve/main/data.zip'
        path = download_url(url, self.root)
        extract_zip(path, self.root)

    def process(self):
        # Read mesh solution into graph structure
        os.makedirs(self.processed_data_dir, exist_ok=True)
        # os.makedirs(self.processed_has_data_dir, exist_ok=True)
        las_data_list = os.listdir(os.path.join(self.raw_data_dir, 'las'))
        has_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has'))
        # has_original_data_list = os.listdir(os.path.join(self.raw_data_dir, 'has_original'))
        data_len = len(las_data_list)
        # mesh_data_list = os.listdir(os.path.join(self.raw_data_dir, 'mesh'))
        # split the list according to the number of processors and process the data in parallel
        num_proc = mp.cpu_count()
        las_data_list = np.array_split(las_data_list, num_proc)
        has_data_list = np.array_split(has_data_list, num_proc)
        # has_original_data_list = np.array_split(has_original_data_list, num_proc)
        
        # organize the data list for each process and combine into pool.map input
        data_list = []
        # progress = mp.Value('i', 0)
        manager = mp.Manager()
        shared_progress_list = manager.list([0] * num_proc)
        for i in range(num_proc):
            data_list.append([self.raw_data_dir, self.processed_data_dir, las_data_list[i], has_data_list[i], i, shared_progress_list])

        # start the progress bar
        progress_thread = Thread(target=update_progress, args=(shared_progress_list, data_len))
        progress_thread.start()

        # start the processes
        with mp.Pool(num_proc) as pool:
            results = [pool.apply_async(process_file_list, args=([data_list[i]])) for i in range(num_proc)]

            for result in results:
                result.get()

        # stop the progress bar
        progress_thread.join()

        # merge the data
        input_file = [os.path.join(self.processed_data_dir, 'data_{}.h5'.format(i)) for i in range(num_proc)]
        # input_file_has = [os.path.join(self.processed_has_data_dir, 'data_{}.h5'.format(i)) for i in range(num_proc)]
        output_file = os.path.join(self.processed_data_dir, 'data.h5')
        # output_file_has = os.path.join(self.processed_has_data_dir, 'data.h5')
        merge_hdf5_files(input_files=input_file, output_file=output_file)
        # self.merge_hdf5_files(input_file_has, output_file_has)
        # redo data list
        
    def transform(self, data):
        if self.transforms == 'error_estimation':
            data.y = data.y - data.x
        if self.transforms == 'normalize':
            # normalize the data layer-wise via gaussian distribution
            data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
            data.y = (data.y - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
        return data

    def get(self, idx):
        data_name = self.data_list[idx]
        str1, str2, str3 = data_name.split('_')
        mesh_name = str1 + '_' + str2
        with h5py.File(os.path.join(self.processed_data_dir, 'data.h5'), 'r') as f:
            grp = f[mesh_name]
            grp_time = grp[str3]
            grp_las = grp_time['las']
            grp_has = grp_time['has']
            las_data_dict = {key: torch.tensor(grp_las[key]) for key in grp_las.keys()}
            has_data_dict = {key: torch.tensor(grp_has[key]) for key in grp_has.keys()}
            data_l = Data.from_dict(las_data_dict)
            data_h = Data.from_dict(has_data_dict)

        if self.transforms is not None:
            data_l = self.transform(data_l)
        return data_l, data_h

    def get_eval(self, idx):
        data_name = self.data_list[idx]
        str1, str2, str3 = data_name.split('_')
        mesh_name = str1 + '_' + str2
        with h5py.File(os.path.join(self.processed_data_dir, 'data.h5'), 'r') as f:
            grp = f[mesh_name]
            grp_time = grp[str3]
            grp_las = grp_time['las']
            grp_has = grp_time['has']
            las_data_dict = {key: torch.tensor(grp_las[key]) for key in grp_las.keys()}
            has_data_dict = {key: torch.tensor(grp_has[key]) for key in grp_has.keys()}
            data_l = Data.from_dict(las_data_dict)
            data_h = Data.from_dict(has_data_dict)

        if self.transforms is not None:
            data_l = self.transform(data_l)
        return data_l, data_h, data_name


class MegaFlow2DSubset(MegaFlow2D):
    """
    This subset splits the entire dataset into 40 subsets, which is initialized via indices.
    """
    def __init__(self, root, indices, transform=None):
        raise NotImplementedError
