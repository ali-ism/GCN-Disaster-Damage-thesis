import json
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EdgeSAGEConv(SAGEConv):
    def _init__(self, *args, edge_dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_lin = Linear(edge_dim, self.out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        self._edge_attr = edge_attr
        out = super().forward(x, edge_index)
        self._edge_attr = None
        return out

    def message(self, x_j, edge_weight):
        return (x_j + self.edge_lin(self._edge_attr)) * edge_weight.view(-1, 1)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, num_layers=2):
        super(SAGENet, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(EdgeSAGEConv(in_channels=in_channels, edge_dim=edge_dim, out_channels=hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(EdgeSAGEConv(in_channels=hidden_channels, edge_dim=edge_dim, out_channels=hidden_channels))
        self.convs.append(EdgeSAGEConv(in_channels=hidden_channels, edge_dim=edge_dim, out_channels=out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, batch):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            edge_attr = batch.edge_attr[e_id].to(device)
            x = self.convs[i]((x, x_target), edge_index, edge_attr)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.sigmoid(x)

    def inference(self, x_all, subgraph_loader, batch):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                edge_attr = batch.edge_attr[e_id].to(device)
                x = self.convs[i]((x, x_target), edge_index, edge_attr)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all