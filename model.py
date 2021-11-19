import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Sequential
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv
from torch_sparse import SparseTensor
from torchvision.models import resnet34


class SiameseEncoder(Module):
    def __init__(self, diff: bool=True) -> None:
        super().__init__()
        self.diff = diff
        self.model = resnet34(pretrained=True, progress=False)
        self.model = Sequential(*(list(self.model.children())[:-1]))
    
    def forward(self, x: Tensor) -> torch.Tensor:
        x = x.reshape((-1, 6, 128, 128))
        x1 = x[:,:3,:,:]
        x2 = x[:,3:,:,:]
        print(f'x1: {x1.dtype}')
        print(f'x2: {x2.dtype}')
        x1 = self.model(x1)
        x2 = self.model(x2)
        if self.diff:
            x = torch.add(input=x1, other=x2, alpha=-1)
        else:
            x = torch.cat((x1, x2), dim=1)
        return x.flatten(start_dim=1)
    
    @torch.no_grad()
    def get_output_shape(self) -> int:
        x = torch.rand(98304)
        x = self.forward(x)
        return x.shape[1]

class SiameseNet(Module):
    def __init__(self, hidden_channels: int, num_classes: int, dropout_rate: float, diff: bool=True) -> None:
        super().__init__()
        
        self.dropout_rate = dropout_rate

        self.encoder = SiameseEncoder(diff)
        self.fc1 = Linear(self.encoder.get_output_shape(), hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.out = Linear(hidden_channels, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
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
    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
        dropout_rate: float,
        diff: bool=True,
        num_meta_features: int=0) -> None:
        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_meta_features = num_meta_features
        self.node_encoder = SiameseEncoder(diff)
        self.convs = ModuleList()
        self.layer_norms = ModuleList()
        input_dim = self.node_encoder.get_output_shape()
        if num_meta_features:
            input_dim += num_meta_features
        self.convs.append(GCNConv(input_dim, hidden_channels, cached=True, normalize=False))
        self.layer_norms.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
            self.layer_norms.append(LayerNorm(hidden_channels))
        self.out = GCNConv(hidden_channels, num_classes, cached=True, normalize=False)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.num_meta_features:
            x = torch.cat([self.node_encoder(x[:,:-2]), x[:,-2:]], dim=1)
        else:
            x  = self.node_encoder(x)
        for layer_norm, conv in zip(self.layer_norms, self.convs):
            x = conv(x, adj_t)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x, adj_t)
        return F.log_softmax(x, dim=1)


class CNNSage(Module):
    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
        dropout_rate: float,
        diff: bool=True) -> None:
        super().__init__()

        self.dropout_rate = dropout_rate
        self.node_encoder = SiameseEncoder(diff)
        self.convs = ModuleList()
        self.layer_norms = ModuleList()
        self.convs.append(SAGEConv(self.node_encoder.get_output_shape(), hidden_channels))
        self.layer_norms.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.layer_norms.append(LayerNorm(hidden_channels))
        self.out = SAGEConv(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x= self.node_encoder(x)
        for layer_norm, conv in zip(self.layer_norms, self.convs):
            x = conv(x, edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x, edge_index)
        return F.log_softmax(x, dim=1)