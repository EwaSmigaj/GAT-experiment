import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph_creation as gc
import cross_validation as cv
from gnn import SimpleGNN
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F



def train(model, optimizer, data, mask):


    model.train()
    optimizer.zero_grad()
    
    x_dict = model(data.x_dict, data.edge_index_dict)
    
    edge_index = data[('user', 'transacts', 'merchant')].edge_index[:, mask]
    src = x_dict['user'][edge_index[0]]
    dst = x_dict['merchant'][edge_index[1]]
    edge_emb = torch.cat([src, dst], dim=-1)
    
    pred = model.lin(edge_emb).squeeze()
    labels = data[('user', 'transacts', 'merchant')].edge_label[mask].float()
    
    # Check for NaN
    if torch.isnan(pred).any():
        print("NaN in predictions during training!")
        return float('nan')
    
    loss = F.binary_cross_entropy_with_logits(pred, labels)
    
    if torch.isnan(loss):
        print("NaN in loss!")
        return float('nan')
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        edge_index = data[('user', 'transacts', 'merchant')].edge_index[:, mask]
        src = x_dict['user'][edge_index[0]]
        dst = x_dict['merchant'][edge_index[1]]
        edge_emb = torch.cat([src, dst], dim=-1)
        
        pred = model.lin(edge_emb).squeeze()
        labels = data[('user', 'transacts', 'merchant')].edge_label[mask].float()

                # Debug prints
        print(f"Pred has NaN: {torch.isnan(pred).any()}")
        print(f"Labels has NaN: {torch.isnan(labels).any()}")
        print(f"Pred range: [{pred.min():.2f}, {pred.max():.2f}]")
        print(f"Labels unique: {labels.unique()}")
        
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        acc = (pred_binary == labels).float().mean()
        
    return loss.item(), acc.item()


def main():
    """Główny punkt wejścia do przetwarzania danych i treningu."""
    
    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")

    """
    GRAPH CREATION
    """
    users, merchants, transactions = gc.load_dataframes()

    # Delete String Columns (so far)
    merchants = gc.handle_string_cols_merchant(merchants)
    transactions = gc.handle_string_cols_edge(transactions)
    users = gc.handle_string_cols_users(users)

    users, merchants, transactions = gc.map_and_clean_ids(users, merchants, transactions)

    data: HeteroData = gc.process_node_features(users, merchants)
    data = gc.process_edge_features(data, transactions)
    data = gc.add_reverse_edges(data)

    """
    TRAINING
    """

    # Main loop
    n_splits = 4
    n_repetitions = 10
    results = []

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, n_splits, 0.6, 0.2)):
        print(f"Fold {fold_idx}:")
        print(f"  Train: {train_mask.sum().item()} edges")
        print(f"  Val: {val_mask.sum().item()} edges")
        print(f"  Test: {test_mask.sum().item()} edges")

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, n_splits, 0.6, 0.2)):
        print(f"\n=== Fold {fold_idx} ===")
        fold_results = []
        
        for rep in range(n_repetitions):
            model = SimpleGNN(hidden_channels=64, out_channels=1, seed=rep)
            model = to_hetero(model, data.metadata(), aggr='sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(100):
                loss = train(model, optimizer, data, train_mask)
            
            test_loss, test_acc = test(model, data, test_mask)
            fold_results.append(test_acc)
            print(f"Rep {rep}: Test Acc = {test_acc:.4f}")
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        print(f"Fold {fold_idx} - Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
        results.append(fold_results)

    print(f"\n=== Overall Results ===")
    print(f"Mean Acc across all folds: {np.mean(results):.4f}")


"""

"""

# print(data)

# model = HeteroGNN(data.metadata(), HIDDEN_CHANNELS, num_edge_features, data).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# class_weights = calculate_class_weights(transactions, device)

# ## 3. Trening i Ewaluacja
# run_training_loop(model, data, optimizer, class_weights)
# final_evaluation(model, data)


if __name__ == '__main__':
    main()