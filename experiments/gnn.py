import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, to_hetero
import random


class SimpleGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, seed=0):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    

class ImprovedGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3, dropout=0.3, seed=0):
        super().__init__()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Edge classifier - individual layers with explicit names
        self.edge_lin1 = torch.nn.Linear(4 * hidden_channels, 2 * hidden_channels)
        self.edge_lin2 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.edge_lin3 = torch.nn.Linear(hidden_channels, out_channels)
        self.edge_relu = torch.nn.ReLU()
        self.edge_dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = x_new.relu()
            x_new = self.dropout(x_new)
            
            if i > 0:
                x_new = x_new + x
            x = x_new
        
        return x
    


