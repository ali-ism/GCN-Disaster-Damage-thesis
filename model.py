import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SAGEConvWithEdges(torch.nn.Module):
    """
    This is an implementation of the GraphSage convolution that also takes into account edge features.
    Source:
        https://github.com/kkonevets/geo_detection/blob/9421a591123c380a1f232b6bff598cae8ff29a23/sage_conv.py
    """
    def __init__(self, in_channels, in_edge_channels, out_channels):
        super(SAGEConvWithEdges, self).__init__()

        self.node_mlp_rel = Linear(in_channels + in_edge_channels, out_channels)

    def forward(self, x, res_size, edge_index, edge_attr):
        row, col = edge_index
        x_row = x[row]

        edge_attr = torch.cat([x_row, edge_attr], 1)

        edge_attr = F.normalize(edge_attr)

        x = scatter(edge_attr, col, dim=0, dim_size=res_size, reduce="mean")
        x = self.node_mlp_rel(x)
        x = F.normalize(x)
        return x
    
    def reset_parameters(self):
        for l in self.modules():
            if type(l) == torch.nn.Linear:
                l.reset_parameters()

    def __repr__(self):
        return ' '.join(
            str([l.in_features, l.out_features]) for l in self.modules()
            if type(l) == torch.nn.Linear)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, in_edge_channels, hidden_channels, out_channels, num_layers):
        super(SAGENet, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConvWithEdges(in_channels, in_edge_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all