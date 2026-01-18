import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, to_hetero, GATConv, Linear, GATv2Conv, LayerNorm
import random


class SimpleGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=None):
        super().__init__()

        if seed is not None: torch.manual_seed(seed)

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


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

        
class AdvancedGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=None, edge_dim=26):
        super().__init__()

        if seed is not None: torch.manual_seed(seed)
        
        # Layer 1
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, heads=4, 
                               edge_dim=edge_dim, concat=True, add_self_loops=False)
        # Skip connection for Layer 1 to match the output dimension (hidden_channels * 4)
        self.skip1 = Linear(-1, hidden_channels * 4)
        self.ln1 = LayerNorm(hidden_channels * 4)
        
        # Layer 2
        self.conv2 = GATv2Conv(hidden_channels * 4, out_channels, heads=1, 
                               edge_dim=edge_dim, concat=False, add_self_loops=False)
        # Skip connection for Layer 2 to match out_channels
        self.skip2 = Linear(hidden_channels * 4, out_channels)
        self.ln2 = LayerNorm(out_channels)
        
        self.dropout = torch.nn.Dropout(0.3)
        self.elu = torch.nn.ELU()

    def forward(self, x, edge_index, edge_attr=None):
        # First Layer + Skip
        # We add the result of the GAT layer to the transformed original features
        x_orig = x
        x = self.conv1(x, edge_index, edge_attr)
        x = x + self.skip1(x_orig) 
        x = self.ln1(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # Second Layer + Skip
        x_orig = x
        x = self.conv2(x, edge_index, edge_attr)
        x = x + self.skip2(x_orig)
        x = self.ln2(x)
        
        return x
    

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

