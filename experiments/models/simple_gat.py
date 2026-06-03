import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, Linear, GATv2Conv, LayerNorm
import random

class SimpleGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=None):
        super().__init__()

        if seed is not None: torch.manual_seed(seed)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

