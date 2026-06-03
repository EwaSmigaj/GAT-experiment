import dgl
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear


class SimpleGAT2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=None):
        super().__init__()
        if seed is not None: torch.manual_seed(seed)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, hg, h_dict):
        # konwertuj DGL heterograf do homogenicznego
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata='h')
            x = g.ndata['h']
            edge_index = torch.stack(g.edges(), dim=0)

        # PyG GAT na homogenicznym grafie
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)

        # z powrotem do h_dict
        h_dict_out = {}
        for i, ntype in enumerate(hg.ntypes):
            mask = (g.ndata['_TYPE'] == i)
            h_dict_out[ntype] = x[mask]

        return h_dict_out