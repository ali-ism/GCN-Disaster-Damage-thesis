import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv


class SageNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SageNet, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        return x