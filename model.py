import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Linear, LayerNorm, ReLU
from torchvision.models import resnet34
from torch_geometric.nn import GCNConv, GENConv, DeepGCNLayer, SAGEConv, BatchNorm


class GCN(Module):
    def __init__(self,
                num_node_features:int,
                hidden_channels: int,
                num_classes: int,
                num_layers: int,
                dropout_rate: float):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.out = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        ###########################
        if edge_attr is not None:
            edge_attr = edge_attr[:,0]
        ###########################
        for batch_norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class DeeperGCN(Module):
    def __init__(self,
                num_node_features: int,
                num_edge_features: int,
                hidden_channels: int,
                num_classes: int,
                num_layers: int,
                dropout_rate: float,
                msg_norm: bool):
        super().__init__()

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
        ##########################################
        if edge_attr is not None:
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


class SiameseEncoder(Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True, progress=False)
        self.model = Sequential(*(list(self.model.children())[:-1]))
    
    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, 6, 128, 128))
        x1 = x[:,:3,:,:]
        x2 = x[:,3:,:,:]
        x1 = self.model(x1)
        x2 = self.model(x2)
        x = torch.add(input=x1, other=x2, alpha=-1)
        return x.flatten(start_dim=1)

class SiameseClf(Module):
    def __init__(self, hidden_channels: int, num_classes: int, dropout_rate: float):
        super().__init__()
        
        self.dropout_rate = dropout_rate

        self.encoder = SiameseEncoder()
        self.fc1 = Linear(512, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.out = Linear(hidden_channels, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


class CNNGCN(Module):
    def __init__(self,
                hidden_channels: int,
                num_classes: int,
                num_layers: int,
                dropout_rate: float):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.node_encoder = SiameseEncoder()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(SAGEConv(512, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.out = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x= self.node_encoder(x)
        for batch_norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x, edge_index)
        return F.log_softmax(x, dim=1)


""" class CNNGCN(Module):
    def __init__(self,
                hidden_channels: int,
                num_classes: int,
                num_layers: int,
                dropout_rate: float):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.node_encoder = SiameseEncoder()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(GCNConv(512, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.out = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        x= self.node_encoder(x)
        ###########################
        if edge_attr is not None:
            edge_attr = edge_attr[:,0]
        ###########################
        for batch_norm, conv in zip(self.batch_norms, self.convs):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.out(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) """