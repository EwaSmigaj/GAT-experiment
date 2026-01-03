import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph_creation as gc
import cross_validation as cv
from gnn import SimpleGNN, ImprovedGNN
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from collections import defaultdict

class TrainingProcessor():
    def __init__(self, model_name, n_reps=2, n_splits=2):
        self.model_name = model_name
        self.n_reps = n_reps
        self.n_splits = n_splits

    # returns results in form of dict
    def cross_validation_training(self, data):

        results = defaultdict(list)

        # print stats 
        for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, self.n_splits, 0.6, 0.2)):
            print(f"Fold {fold_idx}:")
            print(f"  Train: {train_mask.sum().item()} edges")
            print(f"  Val: {val_mask.sum().item()} edges")
            print(f"  Test: {test_mask.sum().item()} edges")

        for rep in range(self.n_reps):
            print(f"\n=== REP {rep} ===")
            fold_results = defaultdict(list)
            
            for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, self.n_splits, 0.6, 0.2)):
                print(f"\n=== Fold {fold_idx} ===")
                if self.model_name == "SimpleGNN":
                    model = SimpleGNN(hidden_channels=64, out_channels=1, seed=rep)
                elif self.model_name == "ImprovedGNN":
                    model = ImprovedGNN(hidden_channels=64, out_channels=1, seed=rep)
                else:
                    model = SimpleGNN(hidden_channels=64, out_channels=1, seed=rep)
                
                model = to_hetero(model, data.metadata())

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                
                for epoch in range(100):
                    loss = self._train(model, optimizer, data, train_mask)
                
                test_loss, tp, tn, fp, fn = self._test(model, data, test_mask)
                
                print(f"loss = {loss}, tp = {tp}, tn = {tn}, fp = {fp}, fn = {fn}")

                fold_results["acc"].append(self._count_acc(tp, tn, fp, fn))
                fold_results["recall"].append(self._count_recall(tp, tn, fp, fn))
                fold_results["precision"].append(self._count_precision(tp, tn, fp, fn))
                fold_results["f1_score"].append(self._count_f1(tp, tn, fp, fn))

                print(fold_results)

                print(f"Rep {fold_idx}: Test Acc = {fold_results["acc"][fold_idx]:.4f}, Recall = {fold_results["recall"][fold_idx]:.4f}, Precision = {fold_results["precision"][fold_idx]:.4f}, F1 = {fold_results["f1_score"][fold_idx]:.4f}" )
            
            mean_acc = np.mean(fold_results["acc"])
            std_acc = np.std(fold_results["acc"])
            print(f"Fold {fold_idx} - Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
            
            for key in fold_results.keys():
                results[key].append(np.mean(fold_results[key]))

        
        print(f"cv training results = {results}")

        return results

    def _train(self, model, optimizer, data, mask):
        model.train()
        optimizer.zero_grad()
        
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        edge_index = data[('user', 'transacts', 'merchant')].edge_index[:, mask]
        src = x_dict['user'][edge_index[0]]
        dst = x_dict['merchant'][edge_index[1]]
        edge_emb = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)

        # Manually apply edge classifier layers
        x = model.edge_lin1(edge_emb)
        x = model.edge_relu(x)
        x = model.edge_dropout(x)
        x = model.edge_lin2(x)
        x = model.edge_relu(x)
        x = model.edge_dropout(x)
        pred = model.edge_lin3(x).squeeze()

        labels = data[('user', 'transacts', 'merchant')].edge_label[mask].float()
        
        # Check for NaN
        if torch.isnan(pred).any():
            print("NaN in predictions during training!")
            return float('nan')

            # Calculate the imbalance ratio
        num_neg = (labels == 0).sum()  # legitimate transactions
        num_pos = (labels == 1).sum()  # fraud transactions
        pos_weight = num_neg / num_pos * 0.5 # e.g., if 99:1 ratio, pos_weight = 99
        
        loss = F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)
        
        if torch.isnan(loss):
            print("NaN in loss!")
            return float('nan')
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


    def _test(self, model, data, mask):
        model.eval()
        with torch.no_grad():

            x_dict = model(data.x_dict, data.edge_index_dict)
            
            edge_index = data[('user', 'transacts', 'merchant')].edge_index[:, mask]
            src = x_dict['user'][edge_index[0]]
            dst = x_dict['merchant'][edge_index[1]]
            edge_emb = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)
        
            # Manually apply edge classifier layers
            x = model.edge_lin1(edge_emb)
            x = model.edge_relu(x)
            x = model.edge_dropout(x)
            x = model.edge_lin2(x)
            x = model.edge_relu(x)
            x = model.edge_dropout(x)
            pred = model.edge_lin3(x).squeeze()

            labels = data[('user', 'transacts', 'merchant')].edge_label[mask].float()

                    # Debug prints
            print(f"Pred has NaN: {torch.isnan(pred).any()}")
            print(f"Labels has NaN: {torch.isnan(labels).any()}")
            print(f"Pred range: [{pred.min():.2f}, {pred.max():.2f}]")
            print(f"Labels unique: {labels.unique()}")
            
            # Calculate the imbalance ratio
            num_neg = (labels == 0).sum()  # legitimate transactions
            num_pos = (labels == 1).sum()  
            pos_weight = num_neg / num_pos  

            print(f"pod_weight = {pos_weight}")

            # Apply weighted loss
            loss = F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)

            loss = F.binary_cross_entropy_with_logits(pred, labels)
            pred_binary = (torch.sigmoid(pred) > 0.7)
            
                    # Calculate confusion matrix components
            tp = ((pred_binary == 1) & (labels == 1)).sum().item()
            tn = ((pred_binary == 0) & (labels == 0)).sum().item()
            fp = ((pred_binary == 1) & (labels == 0)).sum().item()
            fn = ((pred_binary == 0) & (labels == 1)).sum().item()

            return loss.item(), tp, tn, fp, fn


    def _count_acc(self, tp, tn, fp, fn):
        total = tp + tn + fp + fn
        if total == 0:
            return 0
        return (tp + tn) / total

    def _count_precision(self, tp, tn, fp, fn):
        if (tp + fp) == 0:
            return 0
        return tp / (tp + fp)

    def _count_recall(self, tp, tn, fp, fn):
        if (tp + fn) == 0:
            return 0
        return tp / (tp + fn)

    def _count_f1(self, tp, tn, fp, fn):
        precision = self._count_precision(tp, tn, fp, fn)
        recall = self._count_recall(tp, tn, fp, fn)
        if (precision + recall) == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
