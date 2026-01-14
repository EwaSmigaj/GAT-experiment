import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph_creation as gc
import cross_validation as cv
from gnn import SimpleGNN, ImprovedGNN, SimpleGAT
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from collections import defaultdict
from file_logger import log

class TrainingProcessor():
    def __init__(self, model_name, n_reps=15, n_splits=5):
        self.model_name = model_name
        self.n_reps = n_reps
        self.n_splits = n_splits
        self.predictor = EdgePredictor(node_out_channels=64, edge_attr_channels=22)

    # returns results in form of dict
    def cross_validation_training(self, data, balanced_sampling = False):

        results = defaultdict(list)

        # print stats 
        for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, self.n_splits, 0.6, 0.2)):
            print(f"Fold {fold_idx}:")
            print(f"  Train: {train_mask.sum().item()} edges")
            print(f"  Val: {val_mask.sum().item()} edges")
            print(f"  Test: {test_mask.sum().item()} edges")

        for rep in range(self.n_reps):

            print(f"\n=== REP {rep} ===")
            log("=== REP " + str(rep) + " ===")
            
            fold_results = defaultdict(list)
            
            for fold_idx, (train_mask, val_mask, test_mask) in enumerate(cv.split(data, self.n_splits, 0.6, 0.2)):
                print(f"\n=== Fold {fold_idx} ===")

                model = self._pick_model(rep, data)
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(self.predictor.parameters()), 
                    lr=0.01
                )
                
                for epoch in range(100):
                    loss = self._train(model, optimizer, data, train_mask, balanced_sampling)
                
                test_loss, tp, tn, fp, fn = self._test(model, data, test_mask)
                
                log(f"loss = {loss}, tp = {tp}, tn = {tn}, fp = {fp}, fn = {fn}")

                fold_results["acc"].append(self._count_acc(tp, tn, fp, fn))
                fold_results["recall"].append(self._count_recall(tp, tn, fp, fn))
                fold_results["precision"].append(self._count_precision(tp, tn, fp, fn))
                fold_results["f1_score"].append(self._count_f1(tp, tn, fp, fn))

                log(fold_results)

                log(f"Rep {fold_idx}: Test Acc = {fold_results["acc"][fold_idx]:.4f}, Recall = {fold_results["recall"][fold_idx]:.4f}, Precision = {fold_results["precision"][fold_idx]:.4f}, F1 = {fold_results["f1_score"][fold_idx]:.4f}" )
            
            mean_acc = np.mean(fold_results["acc"])
            std_acc = np.std(fold_results["acc"])
            log(f"Fold {fold_idx} - Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
            
            for key in fold_results.keys():
                results[key].append(np.mean(fold_results[key]))

        
        print(f"cv training results = {results}")
        log(results)

        return results


    def _train(self, model, optimizer, data, mask, balanced_sampling):
        model.train()
        self.predictor.train()
        optimizer.zero_grad()

        current_et = ('user', 'transacts', 'merchant')

        # 1. Uzyskanie indeksów krawędzi
        if balanced_sampling:
            # --- WERSJA ULEPSZONA: Balanced Sampling ---
            all_labels = data[current_et].edge_label[mask]
            all_indices = torch.where(mask)[0]
            
            pos_idx = all_indices[all_labels == 1]
            neg_idx = all_indices[all_labels == 0]

            # Losujemy tyle samo negatywnych przykładów, co pozytywnych (ratio 1:1)
            if neg_idx.size(0) > pos_idx.size(0):
                perm = torch.randperm(neg_idx.size(0))[:pos_idx.size(0)]
                sampled_neg_idx = neg_idx[perm]
                train_batch_idx = torch.cat([pos_idx, sampled_neg_idx])
            else:
                train_batch_idx = all_indices
        else:
            # --- WERSJA PO STAREMU: Cała maska ---
            train_batch_idx = torch.where(mask)[0]

        # 2. Forward pass (Embeddingi węzłów)
        h_dict = model(data.x_dict, data.edge_index_dict)

        # 3. Wyciągnięcie danych dla wybranych krawędzi
        edge_index = data[current_et].edge_index[:, train_batch_idx]
        edge_attr = data[current_et].edge_attr[train_batch_idx]
        row, col = edge_index
        labels = data[current_et].edge_label[train_batch_idx].float()

        # 4. Predykcja
        if self.model_name == "ImprovedGNN":
            out = self.predictor(h_dict['user'][row], h_dict['merchant'][col], edge_attr)
        else:
            out = (h_dict['user'][row] * h_dict['merchant'][col]).sum(dim=-1)

        # 5. Strata (Loss)
        if balanced_sampling:
            # Przy zbalansowanych danych nie potrzebujemy wag
            loss = F.binary_cross_entropy_with_logits(out, labels)
        else:
            # Przy niezbalansowanych danych używamy wag
            loss = F.binary_cross_entropy_with_logits(out, labels, pos_weight=self._calc_pos_weight(labels))

        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test(self, model, data, mask):
        model.eval()
        self.predictor.eval() # Set both to eval mode

        # 1. Get Node Embeddings
        h_dict = model(data.x_dict, data.edge_index_dict)
        
        # 2. Get Edge Data
        current_et = ('user', 'transacts', 'merchant')
        edge_index = data[current_et].edge_index[:, mask]
        edge_attr = data[current_et].edge_attr[mask]
        row, col = edge_index
        
        # 3. Predict using the same logic as training
        if self.model_name == "ImprovedGNN":
            z_src = h_dict['user'][row]
            z_dst = h_dict['merchant'][col]
            out = self.predictor(z_src, z_dst, edge_attr)
        else:
            out = (h_dict['user'][row] * h_dict['merchant'][col]).sum(dim=-1)

        labels = data[current_et].edge_label[mask].float()

        # 4. Metrics
        loss = F.binary_cross_entropy_with_logits(out, labels)
        
        # Use sigmoid to get probability, then threshold
        # Tip: Try thresholding at 0.7 or 0.8 if precision is still low
        pred_binary = (out.sigmoid() > 0.8)
        
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

    def _pick_model(self, rep, data):
        if self.model_name == "SimpleGNN":
            model = SimpleGNN(hidden_channels=64, out_channels=64, seed=rep)
        elif self.model_name == "ImprovedGNN":
            model = ImprovedGNN(hidden_channels=64, out_channels=64, seed=rep)
        elif self.model_name == "SimpleGAT":
            model = SimpleGAT(hidden_channels=64, out_channels=64, seed=rep)
        else:
            model = SimpleGNN(hidden_channels=64, out_channels=64, seed=rep)
        
        return to_hetero(model, data.metadata())

    def _calc_pos_weight(self, labels):
        num_neg = (labels == 0).sum()  
        num_pos = (labels == 1).sum()  
        pos_weight = num_neg / num_pos  
        return pos_weight


class EdgePredictor(torch.nn.Module):
    def __init__(self, node_out_channels, edge_attr_channels):
        super().__init__()
        # Input: src_node_emb + dst_node_emb + edge_features
        input_dim = (node_out_channels * 2) + edge_attr_channels
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, z_src, z_dst, edge_attr):
        x = torch.cat([z_src, z_dst, edge_attr], dim=-1)
        return self.lin(x).view(-1)