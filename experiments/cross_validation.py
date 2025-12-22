import torch
import pandas as pd
from torch_geometric.data import HeteroData
from typing import List, Tuple, Optional

        
def split(data, n_splits, train_ratio, val_ratio):
    num_edges = data[('user', 'transacts', 'merchant')].edge_index.size(1)
    window_size = 1.0 / n_splits
    
    for i in range(n_splits):
        start = int(i * window_size * num_edges)
        end = int((i + 1) * window_size * num_edges)
        window_edges = end - start
        
        train_end = start + int(window_edges * train_ratio)
        val_end = train_end + int(window_edges * val_ratio)
        test_end = end
        
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        train_mask[start:train_end] = True
        val_mask[train_end:val_end] = True
        test_mask[val_end:test_end] = True
        
        yield train_mask, val_mask, test_mask
