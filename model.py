import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer, SplineConv, BatchNorm


class DeeperGCN(Module):
    def __init__(self,
                 num_node_features,
                 num_edge_features,
                 hidden_channels,
                 num_classes,
                 num_layers,
                 dropout_rate):
        super(DeeperGCN, self).__init__()

        self.dropout_rate = dropout_rate
        self.node_encoder = Linear(num_node_features, hidden_channels)
        self.edge_encoder = Linear(num_edge_features, hidden_channels)
        self.layers = ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout_rate, ckpt_grad=i % 3)
            self.layers.append(layer)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class SplineNet(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 hidden_channels,
                 num_classes,
                 num_layers,
                 dropout_rate):
        super(SplineNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(SplineConv(num_node_features, hidden_channels, dim=1, kernel_size=2))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SplineConv(hidden_channels, hidden_channels, dim=1, kernel_size=2))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.out = SplineConv(hidden_channels, num_classes, dim=1, kernel_size=2)


    def forward(self, x, edge_index, edge_attr):
        for batch_norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)