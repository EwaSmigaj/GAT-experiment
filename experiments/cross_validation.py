import torch
import pandas as pd
from torch_geometric.data import HeteroData
from typing import List, Tuple, Optional
import numpy as np

        
def split(data, n_splits, train_ratio, val_ratio, neg_to_pos_ratio=10, seed=None):
    num_edges = data[('user', 'transacts', 'merchant')].edge_index.size(1)
    labels = data[('user', 'transacts', 'merchant')].edge_label
    
    window_size = 1.0 / n_splits

    if seed is None: seed = 1 

    torch.manual_seed(seed)
    np.random.seed(seed)

    output = []
    
    for i in range(n_splits):
        # 1. Definiowanie zakresów okna
        start = int(i * window_size * num_edges)
        end = int((i + 1) * window_size * num_edges)
        window_edges = end - start
        
        train_end = start + int(window_edges * train_ratio)
        val_end = train_end + int(window_edges * val_ratio)
        
        # 2. Tworzenie bazowych masek
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask[train_end:val_end] = True
        test_mask[val_end:end] = True

        # 3. BALANSOWANIE MASKI TRENINGOWEJ
        # Wyciągamy indeksy potencjalnie treningowe
        raw_train_indices = torch.arange(start, train_end)
        train_labels = labels[raw_train_indices]

        pos_train_idx = raw_train_indices[train_labels == 1]
        neg_train_idx = raw_train_indices[train_labels == 0]

        # Obliczamy ile negatywnych przykładów chcemy zostawić
        n_pos = pos_train_idx.size(0)
        n_neg_to_keep = n_pos * neg_to_pos_ratio

        if neg_train_idx.size(0) > n_neg_to_keep:
            # Losujemy podzbiór negatywnych indeksów
            perm = torch.randperm(neg_train_idx.size(0))[:n_neg_to_keep]
            sampled_neg_idx = neg_train_idx[perm]
            # Łączymy pozytywne z wybranymi negatywnymi
            final_train_indices = torch.cat([pos_train_idx, sampled_neg_idx])
        else:
            final_train_indices = raw_train_indices

        # Tworzymy finalną maskę treningową
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[final_train_indices] = True
        
        output.append((train_mask, val_mask, test_mask))
    return output
