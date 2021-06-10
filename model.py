import json
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool

with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PNANet(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_dim, deg):
        super(PNANet, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=node_dim, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_dim, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.mlp = Sequential(Linear(hidden_channels, hidden_channels//2), ReLU(),
                              Linear(hidden_channels//2, hidden_channels//4), ReLU(),
                              Linear(hidden_channels//4, out_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return F.sigmoid(self.mlp(x))