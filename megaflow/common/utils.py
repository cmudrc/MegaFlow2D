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
    las_save_dir = data_list[1]
    has_save_dir = data_list[2]
    las_data_list = data_list[3]
    has_data_list = data_list[4]
    # has_original_data_list = data_list[5]
    index = data_list[5]
    shared_progress_list = data_list[6]
    processed_file_count = 0
    with h5py.File(os.path.join(las_save_dir, 'data_{}.h5'.format(index)), 'a') as las_h5_file:
        with h5py.File(os.path.join(has_save_dir, 'data_{}.h5'.format(index)), 'a') as has_h5_file:
            for las_data_name, has_data_name in zip(las_data_list, has_data_list):
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

                node_data_list = np.concatenate((node_data_x, node_data_y, node_data_p), axis=1)
                val_data_list = np.concatenate((val_data_x, val_data_y, val_data_p), axis=1)
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
                node_pos_list = np.concatenate((node_pos_x, node_pos_y), axis=1)
                # for j in range(len(mesh_data['x'])):
                #     node_pos[0] = mesh_data['x'][j]
                #     node_pos[1] = mesh_data['y'][j]

                #     if j == 0:
                #         node_pos_list = np.array([node_pos])
                #     else:
                #         node_pos_list = np.append(node_pos_list, np.array([node_pos]), axis=0)

                node_pos_list = torch.tensor(node_pos_list, dtype=torch.float)

                data_las = Data(x=node_data_list, y=val_data_list, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, pos=node_pos_list)
                # print("las data process done")
                # process has graph
                has_data_original = np.load(os.path.join(raw_data_dir, 'has_original', has_data_name))
                mesh_data = np.load(os.path.join(raw_data_dir, 'mesh', 'has', mesh_name))
                # node_data = np.zeros(3)
                node_data_x = has_data_original['ux']
                node_data_y = has_data_original['uy']
                node_data_p = has_data_original['p']

                node_data_list = np.concatenate((node_data_x, node_data_y, node_data_p), axis=1)
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
                node_pos_list = np.concatenate((node_pos_x, node_pos_y), axis=1)
                # node_pos = np.zeros(2)
                # for j in range(len(mesh_data['x'])):
                #     node_pos[0] = mesh_data['x'][j]
                #     node_pos[1] = mesh_data['y'][j]

                #     if j == 0:
                #         node_pos_list = np.array([node_pos])
                #     else:
                #         node_pos_list = np.append(node_pos_list, np.array([node_pos]), axis=0)

                node_pos_list = torch.tensor(node_pos_list, dtype=torch.float)

                data_has = Data(x=node_data_list, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, pos=node_pos_list)
                # print("has data process done")
                data_name = str1 + '_' + str2 + '_' + str4

                grp_las = las_h5_file.create_group(data_name)
                for key, item in data_las:
                    grp_las.create_dataset(key, data=item.numpy(), compression="gzip", compression_opts=7)
                # las_h5_file.flush()

                grp_has = has_h5_file.create_group(data_name)
                for key, item in data_has:
                    grp_has.create_dataset(key, data=item.numpy(), compression="gzip", compression_opts=7)
                # has_h5_file.flush()
                # print("data save done")
                # with progress.get_lock():
                #     progress.value += 1
                processed_file_count += 1
                # update progress every 10 files
                if processed_file_count % 10 == 0 and processed_file_count > 0:
                    shared_progress_list.append("update")


def update_progress(shared_progress_list, total_data):
    with tqdm(total=total_data) as pbar:
        while len(shared_progress_list) * 10 < total_data - 10:
            current_len = len(shared_progress_list) * 10
            pbar.update(current_len - pbar.n)
            time.sleep(1)