import torch
import torch_geometric.nn as nn
from torch_geometric.nn.unpool import knn_interpolate

class FlowMLConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_filters):
        super(FlowMLConvolution, self).__init__()
        # self.conv1 = nn.Sequential('x, edge_index', [(nn.GraphConv(in_channels, num_filters[0]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[0])])
        self.conv1 = nn.Sequential('x, edge_index', [(nn.GraphConv(in_channels, num_filters[0]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)])
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            # self.convs.append(nn.Sequential('x, edge_index', [(nn.GraphConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[i+1])]))
            self.convs.append(nn.Sequential('x, edge_index', [(nn.GraphConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)]))
        self.conv2 = nn.GraphConv(num_filters[i+1], out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = nn.global_mean_pool(x, batch)
        return x


class EdgeConv(nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add')
        self.mlp = torch.nn.Sequential(nn.Linear(2*in_channels, 32), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(32), torch.nn.Linear(32, 32), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(32), torch.nn.Linear(32, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j-x_i], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = nn.knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)


class FlowMLError(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FlowMLError, self).__init__()
        # self.convs = FlowMLConvolution(in_channels+1, out_channels, 3, [64, 64, 64])
        
        self.edge_conv1 = DynamicEdgeConv(2, 64)
        self.edge_convs = torch.nn.ModuleList()
        for i in range(2):
            self.edge_convs.append(DynamicEdgeConv(64, 64))
        self.edge_conv2 = DynamicEdgeConv(64, 128)
        self.edge_conv3 = DynamicEdgeConv(192, 1)

        self.conv4 = nn.Sequential('x, edge_index', [(nn.GCNConv(in_channels+1, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)])
        self.convs2 = torch.nn.ModuleList()
        for i in range(3):
            # self.convs.append(nn.Sequential('x, edge_index', [(nn.GraphConv(num_filters[i], num_filters[i+1]), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1), torch.nn.BatchNorm1d(num_filters[i+1])]))
            self.convs2.append(nn.Sequential('x, edge_index', [(nn.GCNConv(64, 64), 'x, edge_index -> x'), torch.nn.LeakyReLU(0.1)]))
        self.conv5 = nn.GCNConv(64, out_channels)

        
    def forward(self, data):
        u, coord, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = self.edge_conv1(coord, batch)
        append = x
        for conv in self.edge_convs:
            x = conv(x, batch)
            torch.cat((append, x), dim=1)
        x = self.edge_conv2(x, batch)
        x = self.edge_conv3(torch.cat([append, x], dim=1), batch)

        u = self.conv4(torch.cat([u, x], dim=1), edge_index)
        for conv in self.convs2:
            u = conv(u, edge_index)
        u = self.conv5(u, edge_index)

        return u
