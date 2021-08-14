import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer, SplineConv, GCNConv, BatchNorm


class DeeperGCN(Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 hidden_channels: int,
                 num_classes: int,
                 num_layers: int,
                 dropout_rate: float,
                 msg_norm: bool):
        super(DeeperGCN, self).__init__()

        self.dropout_rate = dropout_rate
        self.node_encoder = Linear(num_node_features, hidden_channels)
        ################################################################
        self.edge_encoder = Linear(num_edge_features-1, hidden_channels)
        ################################################################
        self.layers = ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, learn_t=True, num_layers=2,
                           norm='layer', msg_norm=msg_norm, learn_msg_scale=msg_norm)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout_rate, ckpt_grad=i % 3)
            self.layers.append(layer)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            return self.forward_no_edge(x, edge_index)
        else:
            ##########################################
            edge_attr = edge_attr[:,0].unsqueeze(dim=1)
            ###########################################
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            x = self.layers[0].conv(x, edge_index, edge_attr)
            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)
            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.lin(x)
            return F.log_softmax(x, dim=1)
    
    def forward_no_edge(self, x, edge_index):
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


""" class SplineNet(Module):
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
        for _ in range(num_layers - 2):
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
        return F.log_softmax(x, dim=1) """


class GCN(Module):
    def __init__(self,
                 num_node_features,
                 hidden_channels,
                 num_classes,
                 num_layers,
                 dropout_rate,
                 fc_output=False):
        super(GCN, self).__init__()

        self.dropout_rate = dropout_rate
        self.fc_output = fc_output
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        if not fc_output:
            self.out = GCNConv(hidden_channels, num_classes)
        else:
            self.fc = Linear(hidden_channels, hidden_channels)
            self.bn = BatchNorm(hidden_channels)
            self.fcout = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            return self.edge_forward(x, edge_index, edge_attr)
        else:
            for batch_norm, conv in zip(self.batch_norms, self.convs):
                x = conv(x, edge_index)
                x = batch_norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if not self.fc_output:
                x = self.out(x, edge_index)
            else:
                x = self.fc(x)
                x = self.bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                x = self.fcout(x)
            return F.log_softmax(x, dim=1)
    
    def edge_forward(self, x, edge_index, edge_attr):
        ###########################
        edge_attr = edge_attr[:,0]
        ###########################
        for batch_norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if not self.fc_output:
            x = self.out(x, edge_index, edge_attr)
        else:
            x = self.fc(x)
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.fcout(x)
        return F.log_softmax(x, dim=1)