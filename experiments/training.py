import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph_creation as gc
import cross_validation as cv
from gnn import SimpleGNN, ImprovedGNN, SimpleGAT, AdvancedGNN
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from collections import defaultdict
from file_logger import log
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class TrainingProcessor():
    def __init__(self, model_name, n_reps=15):
        self.model_name = model_name
        self.n_reps = n_reps
        self.predictor = EdgePredictor(node_out_channels=64, edge_attr_channels=22)

    def cross_validation_training(self, data_masks, data):

        results = defaultdict(list)
        all_epoch_stats = []  # do wykresów

        for rep in range(self.n_reps):
            log(f"\n=== REP {rep} ===")
            fold_results = defaultdict(list)
            fold_confusion_matrices = []
            fold_mcen_aux = []

            for fold_idx, (train_mask, val_mask, test_mask) in enumerate(data_masks):
                log(f"\n--- Rep {rep} | Fold {fold_idx} ---")

                model = self._pick_model(rep, data)
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(self.predictor.parameters()), lr=0.01
                )

                best_val_pr_auc = -1
                best_epoch = -1
                best_model_state = None
                best_predictor_state = None

                # Statystyki per epoka
                epoch_stats = {
                    "loss": [], "val_pr_auc": [], "val_f1": [],
                    "val_precision": [], "val_recall": []
                }

                log("Starting epoch")

                # -------------------------------
                # Trening
                # -------------------------------
                for epoch in range(100):
                    loss, avg_pos_score, avg_neg_score = self._train(model, optimizer, data, train_mask)

                    if epoch % 10 == 0 or epoch == 99:
                        
                        val_loss, val_probs, val_labels = self._test(model, data, val_mask)
                        # Tymczasowy threshold 0.5 dla monitorowania
                        val_metrics = self._eval_with_threshold(val_probs, val_labels, threshold=0.5)
                        pr_auc = average_precision_score(val_labels.numpy(), val_probs.numpy())

                        # Zapis statystyk
                        epoch_stats["loss"].append(val_loss)
                        epoch_stats["val_pr_auc"].append(pr_auc)
                        epoch_stats["val_f1"].append(val_metrics["f1"])
                        epoch_stats["val_precision"].append(val_metrics["precision"])
                        epoch_stats["val_recall"].append(val_metrics["recall"])

                        log(f"epoch {epoch} stats: {epoch_stats}")

                        # Wybór najlepszej epoki
                        if pr_auc > best_val_pr_auc:
                            best_val_pr_auc = pr_auc
                            best_epoch = epoch
                            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                            best_predictor_state = {k: v.cpu() for k, v in self.predictor.state_dict().items()}

                # -------------------------------
                # Najlepsza epoka i treshold
                # -------------------------------
                model.load_state_dict(best_model_state)
                self.predictor.load_state_dict(best_predictor_state)

                _, val_probs, val_labels = self._test(model, data, val_mask)
                best_threshold = self._find_best_threshold(val_labels.numpy(), val_probs.numpy())

                # -------------------------------
                # Test końcowy
                # -------------------------------
                _, test_probs, test_labels = self._test(model, data, test_mask)
                test_metrics = self._eval_with_threshold(test_probs, test_labels, best_threshold)

                fold_confusion_matrices.append(test_metrics["confusion_matrix"])
                fold_mcen_aux.append(test_metrics["mcen"])

                fold_results["acc"].append(test_metrics['acc'])
                fold_results["recall"].append(test_metrics['recall'])
                fold_results["precision"].append(test_metrics['precision'])
                fold_results["f1_score"].append(test_metrics['f1'])

                log(f"Fold {fold_idx} | BEST EPOCH={best_epoch} | PR-AUC(val)={best_val_pr_auc:.4f} | "
                    f"Threshold={best_threshold:.3f} | Test F1={test_metrics['f1']:.4f} | "
                    f"Precision={test_metrics['precision']:.4f} | Recall={test_metrics['recall']:.4f} | "
                    f"MCEN(aux)={test_metrics['mcen']:.4f}")

                # Zapis statystyk per fold do wykresów
                all_epoch_stats.append(epoch_stats)

            # -------------------------------
            # Agregacja wyników foldów
            # -------------------------------
            for key in fold_results.keys():
                results[key].append(np.mean(fold_results[key]))

            cm_total = np.zeros((2, 2), dtype=int)
            for cm in fold_confusion_matrices:
                cm_total += cm

            global_mcen = self._compute_mcen(cm_total)
            aux_mean = np.mean(fold_mcen_aux)
            aux_std = np.std(fold_mcen_aux)

            log(f"MCEN global = {global_mcen:.4f}")
            log(f"MCEN per fold (auxiliary): mean={aux_mean:.4f}, std={aux_std:.4f}")

        # -------------------------------
        # Wykres agregowany po wszystkich rep/fold
        # -------------------------------
        def plot_mean_training_curve(all_epoch_stats, metric="val_f1"):
            min_len = min(len(e[metric]) for e in all_epoch_stats)
            metric_values = np.array([e[metric][:min_len] for e in all_epoch_stats])
            mean_metric = metric_values.mean(axis=0)
            std_metric = metric_values.std(axis=0)
            epochs = len(mean_metric)

            plt.figure(figsize=(8,5))
            plt.plot(range(1, epochs+1), mean_metric, label=f"mean {metric}")
            plt.fill_between(range(1, epochs+1),
                            mean_metric - std_metric,
                            mean_metric + std_metric,
                            alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Epochs (mean ± std across folds and reps)")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Rysowanie wykresów raportowych
        # plot_mean_training_curve(all_epoch_stats, metric="val_f1")
        # plot_mean_training_curve(all_epoch_stats, metric="val_pr_auc")
        # plot_mean_training_curve(all_epoch_stats, metric="loss")

        print(f"cv training results = {results}")
        log(results)

        return results


    def _train(self, model, optimizer, data, mask):
        model.train()
        self.predictor.train()
        optimizer.zero_grad()

        current_et = ('user', 'transacts', 'merchant')

        train_batch_idx = torch.where(mask)[0]

        # 2. Forward pass (Embeddingi węzłów)
        h_dict = model(data.x_dict, data.edge_index_dict)

        # 3. Wyciągnięcie danych dla wybranych krawędzi
        edge_index = data[current_et].edge_index[:, train_batch_idx]
        edge_attr = data[current_et].edge_attr[train_batch_idx]
        row, col = edge_index
        labels = data[current_et].edge_label[train_batch_idx].float()

        # 4. Predykcja
        out = self.predictor(h_dict['user'][row], h_dict['merchant'][col], edge_attr)

        p_weight = torch.tensor([1.0]).to(out.device) 
        loss = F.binary_cross_entropy_with_logits(out, labels, pos_weight=p_weight)
        
        with torch.no_grad():
            probs = torch.sigmoid(out)
            avg_pos_score = probs[labels == 1].mean().item() if (labels == 1).any() else 0.0
            avg_neg_score = probs[labels == 0].mean().item() if (labels == 0).any() else 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(self.predictor.parameters()), max_norm=1.0)
        optimizer.step()

        return loss.item(), avg_pos_score, avg_neg_score

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
        
        z_src = h_dict['user'][row]
        z_dst = h_dict['merchant'][col]
        out = self.predictor(z_src, z_dst, edge_attr)

        labels = data[current_et].edge_label[mask].float()
        loss = F.binary_cross_entropy_with_logits(out, labels)

        probs = out.sigmoid()

        return loss.item(), probs.cpu(), labels.cpu()

    def _find_best_threshold(self, y_true, y_probs):
        best_threshold = 0.5
        best_f1 = 0
        
        thresholds = np.linspace(0, 1, 100)
        
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = t

        # dodatkowe metryki dla logów
        y_pred_best = (y_probs >= best_threshold).astype(int)
        prec = precision_score(y_true, y_pred_best, zero_division=0)
        rec = recall_score(y_true, y_pred_best, zero_division=0)

        log(f"--- Optymalizacja progu ---")
        log(f"Najlepszy próg: {best_threshold:.4f}")
        log(f"F1: {best_f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        return best_threshold


    def _plot_precision_recall(self, y_true, y_probs):
        # 1. Obliczanie punktów krzywej
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        
        # 2. Obliczanie pola pod krzywą (Average Precision)
        ap_score = average_precision_score(y_true, y_probs)
        
        # 3. Znalezienie najlepszego punktu F1 na krzywej
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_t = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        
        return best_t

    def _eval_with_threshold(self, probs, labels, threshold):
        preds = (probs >= threshold).int()
        labels = labels.int()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        acc = self._count_acc(tp, tn, fp, fn)
        prec = self._count_precision(tp, tn, fp, fn)
        rec = self._count_recall(tp, tn, fp, fn)
        f1 = self._count_f1(tp, tn, fp, fn)

        confusion_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])
        mcen = self._compute_mcen(confusion_matrix)

        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "confusion_matrix": confusion_matrix,
            "mcen": mcen
        }

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
            model = AdvancedGNN(hidden_channels=64, out_channels=64, seed=rep)
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

    def _compute_mcen(self, confusion_matrix, eps=1e-12):
        """
        MCEN for binary classification
        confusion_matrix:
            [[TN, FP],
            [FN, TP]]
        """
        cm = confusion_matrix.astype(float)

        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]

        # Row and column sums
        row0 = TN + FP
        row1 = FN + TP
        col0 = TN + FN
        col1 = FP + TP

        # Denominators
        denom0 = row0 + col0
        denom1 = row1 + col1

        cen0 = 0.0
        if FP > 0:
            p0 = FP / (denom0 + eps)
            cen0 = -p0 * np.log(p0 + eps)

        cen1 = 0.0
        if FN > 0:
            p1 = FN / (denom1 + eps)
            cen1 = -p1 * np.log(p1 + eps)

        mcen = 0.5 * (cen0 + cen1)
        return mcen


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