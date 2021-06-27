import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer


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

        return torch.sigmoid(self.lin(x))