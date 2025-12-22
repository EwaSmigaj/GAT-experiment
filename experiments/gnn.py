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

 