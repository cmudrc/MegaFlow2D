from megaflow.dataset.MegaFlow2D import MegaFlow2D
import torch
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    dataset = MegaFlow2D(root='C:/research/data', download=False, split_scheme='mixed', transform=None, pre_transform=None, split_ratio=[0.5, 0.5])

    graph_data = dataset.get(1870)

    print('Graph basic statistics: num_nodes: {}, num_edges: {}, num_node_features: {}, num_edge_features: {}'.\
        format(graph_data.num_nodes, graph_data.num_edges, graph_data.num_node_features, graph_data.num_edge_features))
    
    print('Graph topolgy statistics: is_directed: {}'.format(graph_data.is_directed()))

    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for batch in train_dataloader:
        print(batch)
        break