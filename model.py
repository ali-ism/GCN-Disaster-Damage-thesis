import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, Linear
from torchvision.models import resnet34
from torch_geometric.nn import SAGEConv, BatchNorm


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

class SiameseNet(Module):
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