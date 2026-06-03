import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, Linear, GATv2Conv, LayerNorm
import random

class ImprovedGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=None):
        super().__init__()
        if seed is not None: torch.manual_seed(seed)

        # GNN Layers
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x