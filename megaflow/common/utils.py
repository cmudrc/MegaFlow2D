import os
import time
from datetime import datetime
import torch
from torch_geometric.data import Data
# from model import *
# from dataset import *
import h5py
import numpy as np
from tqdm import tqdm
from queue import Empty


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def process_file_list(data_list):
    raw_data_dir = data_list[0]
    save_dir = data_list[1]
    # has_save_dir = data_list[2]
    las_data_list = data_list[2]
    has_data_list = data_list[3]
    # has_original_data_list = data_list[5]
    index = data_list[4]
    shared_progress_list = data_list[5]
    processed_file_count = 0
    with h5py.File(os.path.join(save_dir, 'data_{}.h5'.format(index)), 'a') as f:
        # with h5py.File(os.path.join(has_save_dir, 'data_{}.h5'.format(index)), 'a') as has_h5_file:
        for las_data_name, has_data_name in zip(las_data_list, has_data_list):
            # process data name into format geometry_index_timestep
            str1, str2, str3, str4 = las_data_name.split('_')
            str4 = str4.split('.')[0]
            mesh_name = str1 + '_' + str2

            # check if the mesh type is in the h5 file, if not, create a group for it
            try:
                grp = f[mesh_name]
                grp_time = grp[str4]
                grp_las = grp_time['las']
                grp_has = grp_time['has']
            except KeyError:
                grp = f.require_group(mesh_name)
                grp_time = grp.require_group(str4)
                grp_las = grp_time.require_group('las')
                grp_has = grp_time.require_group('has')
            
            # process las graph
            las_data = np.load(os.path.join(raw_data_dir, 'las', las_data_name))
            has_data = np.load(os.path.join(raw_data_dir, 'has', has_data_name))

            str1, str2, str3, str4 = las_data_name.split('_')
            str4 = str4.split('.')[0]
            mesh_name = str1 + '_' + str2 + '.npz'
            mesh_data = np.load(os.path.join(raw_data_dir, 'mesh', 'las', mesh_name))
            # node_data = np.zeros(3)
            # val_data = np.zeros(3)

            node_data_x = las_data['ux']
            node_data_y = las_data['uy']
            node_data_p = las_data['p']

            val_data_x = has_data['ux']
            val_data_y = has_data['uy']
            val_data_p = has_data['p']

            node_data_list = np.column_stack((node_data_x, node_data_y, node_data_p))
            val_data_list = np.column_stack((val_data_x, val_data_y, val_data_p))
            # for j in range(len(mesh_data['x'])):
            #     node_data[0] = las_data['ux'][j]
            #     node_data[1] = las_data['uy'][j]
            #     node_data[2] = las_data['p'][j]

            #     val_data[0] = has_data['ux'][j]
            #     val_data[1] = has_data['uy'][j]
            #     val_data[2] = has_data['p'][j]

            #     if j == 0:
            #         node_data_list = np.array([node_data])
            #         val_data_list = np.array([val_data])
            #     else:
            #         node_data_list = np.append(node_data_list, np.array([node_data]), axis=0)
            #         val_data_list = np.append(val_data_list, np.array([val_data]), axis=0)

            node_data_list = torch.tensor(node_data_list, dtype=torch.float)
            val_data_list = torch.tensor(val_data_list, dtype=torch.float)

            # process edge
            edge_index = np.array(mesh_data['edges'])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = np.array(mesh_data['edge_properties'])
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # node_pos = np.zeros(2)
            node_pos_x = mesh_data['x']
            node_pos_y = mesh_data['y']
            node_pos_list = np.column_stack((node_pos_x, node_pos_y))
            # for j in range(len(mesh_data['x'])):
            #     node_pos[0] = mesh_data['x'][j]
            #     node_pos[1] = mesh_data['y'][j]

            #     if j == 0:
            #         node_pos_list = np.array([node_pos])
            #     else:
            #         node_pos_list = np.append(node_pos_list, np.array([node_pos]), axis=0)

            node_pos_list = torch.tensor(node_pos_list, dtype=torch.float)

            # create a python dictionary to store the data
            # data_las = {'x': node_data_list, 'y': val_data_list, 'edge_index': edge_index.t().contiguous(), 'edge_attr': edge_attr, 'pos': node_pos_list}
            data_las = Data(x=node_data_list, y=val_data_list, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, pos=node_pos_list)
            # print("las data process done")
            # process has graph
            has_data_original = np.load(os.path.join(raw_data_dir, 'has_original', has_data_name))
            mesh_data = np.load(os.path.join(raw_data_dir, 'mesh', 'has', mesh_name))
            # node_data = np.zeros(3)
            node_data_x = has_data_original['ux']
            node_data_y = has_data_original['uy']
            node_data_p = has_data_original['p']

            node_data_list = np.column_stack((node_data_x, node_data_y, node_data_p))
            # for j in range(len(mesh_data['x'])):
            #     node_data[0] = has_data_original['ux'][j]
            #     node_data[1] = has_data_original['uy'][j]
            #     node_data[2] = has_data_original['p'][j]

            #     if j == 0:
            #         node_data_list = np.array([node_data])
            #     else:
            #         node_data_list = np.append(node_data_list, np.array([node_data]), axis=0)

            node_data_list = torch.tensor(node_data_list, dtype=torch.float)
            edge_index = np.array(mesh_data['edges'])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = np.array(mesh_data['edge_properties'])
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            node_pos_x = mesh_data['x']
            node_pos_y = mesh_data['y']
            node_pos_list = np.column_stack((node_pos_x, node_pos_y))
            # node_pos = np.zeros(2)
            # for j in range(len(mesh_data['x'])):
            #     node_pos[0] = mesh_data['x'][j]
            #     node_pos[1] = mesh_data['y'][j]

            #     if j == 0:
            #         node_pos_list = np.array([node_pos])
            #     else:
            #         node_pos_list = np.append(node_pos_list, np.array([node_pos]), axis=0)

            node_pos_list = torch.tensor(node_pos_list, dtype=torch.float)

            # create a python dictionary to store the data
            # data_has = {'x': node_data_list, 'edge_index': edge_index.t().contiguous(), 'edge_attr': edge_attr, 'pos': node_pos_list}
            data_has = Data(x=node_data_list, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, pos=node_pos_list)
            # print("has data process done")

            # write las, has data to dset, with key being time step, and data being the data object
            for key, value in data_las:
                grp_las.create_dataset(key, data=value.numpy(), compression="gzip", compression_opts=9, chunks=True)
            for key, value in data_has:
                grp_has.create_dataset(key, data=value.numpy(), compression="gzip", compression_opts=9, chunks=True)
            # has_h5_file.flush()
            # print("data save done")
            # with progress.get_lock():
            #     progress.value += 1
            processed_file_count += 1
            # update progress every 10 files
            if processed_file_count % 10 == 0 and processed_file_count > 0:
                shared_progress_list[index] = processed_file_count
        
        # update progress 
        shared_progress_list[index] = processed_file_count

    # print("process done")
    return 0


def update_progress(shared_progress_list, total_data):
    with tqdm(total=total_data) as pbar:
        while np.sum(np.array(shared_progress_list)) < total_data - 1:
            current_len = np.sum(np.array(shared_progress_list))
            pbar.update(current_len - pbar.n)
            time.sleep(1)


def copy_group(src_group, dst_group):
    for key in src_group.keys():
        src_item = src_group[key]
        if isinstance(src_item, h5py.Group):
            # Create a subgroup in the destination group if it doesn't exist
            if key not in dst_group:
                dst_group.create_group(key)
            dst_subgroup = dst_group[key]
            copy_group(src_item, dst_subgroup)
        else:
            src_group.copy(key, dst_group)


def merge_hdf5_files(input_files, output_file):
    with h5py.File(output_file, 'a') as output_h5:
        for input_file in input_files:
            with h5py.File(input_file, 'r') as input_h5:
                copy_group(input_h5, output_h5)

        # Remove the input files after merging
        for input_file in input_files:
            os.remove(input_file)