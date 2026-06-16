import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_recall_curve
from evaluation.file_logger import log
from models.external.simpleHGN import SimpleHGN


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, labels):
        # reduction='none' aby policzyć wagę dla każdej próbki
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class InputEncoder(nn.Module):
    """Trenowalna projekcja per typ węzła -> wspólny wymiar proj_dim."""

    def __init__(self, hg, proj_dim: int = 32):
        super().__init__()
        self.encoders = nn.ModuleDict({
            ntype: nn.Linear(hg.nodes[ntype].data['h_raw'].shape[1], proj_dim)
            for ntype in hg.ntypes
        })

    def forward(self, hg):
        return {ntype: enc(hg.nodes[ntype].data['h_raw'])
                for ntype, enc in self.encoders.items()}


class TrainingProcessor:

    def __init__(self, model_name, n_reps=3, n_epochs=100, loss_mul=1.0):
        self.model_name = model_name
        self.n_reps     = n_reps
        self.n_epochs   = n_epochs
        self.loss_mul   = loss_mul

    # ──────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────

    def cross_validation_training(self, data_masks, data):
        results = defaultdict(list)

        for rep in range(self.n_reps):
            log(f"\n=== REP {rep} ===")
            fold_results  = defaultdict(list)
            fold_cms      = []
            fold_mcen_aux = []

            for fold_idx, (train_mask, val_mask, test_mask) in enumerate(data_masks):

                labels = data.nodes['transaction'].data['label']

                for mask, name in [(train_mask, 'train'), (val_mask, 'val'), (test_mask, 'test')]:
                    n_pos = labels[mask].sum().item()
                    n_neg = (labels[mask] == 0).sum().item()
                    log(f"  {name:>5}: fraud={n_pos:>6}  non-fraud={n_neg:>8}  ratio={n_pos/(n_neg+1e-9):.4f}")

                log(f"\n--- Rep {rep} | Fold {fold_idx} ---")

                model, encoder = self._build(rep, data)
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(encoder.parameters()), lr=0.01
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3
                )

                # pos_weight raz z train_mask – nie przeliczamy w _eval_step
                train_labels = data.nodes['transaction'].data['label'][train_mask]
                pos_weight   = self._calc_pos_weight(train_labels)

                best_pr_auc      = -1
                best_epoch       = -1
                best_model_state = best_encoder_state = None
                epoch_stats      = defaultdict(list)

                log("Starting epoch")
                for epoch in range(self.n_epochs):
                    self._train_step(model, encoder, optimizer, data, train_mask, pos_weight)

                    if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                        val_loss, val_probs, val_labels = self._eval_step(
                            model, encoder, data, val_mask, pos_weight)
                        pr_auc = average_precision_score(val_labels.numpy(), val_probs.numpy())
                        m      = self._eval_with_threshold(val_probs, val_labels, 0.5)

                        epoch_stats["loss"].append(val_loss)
                        epoch_stats["val_pr_auc"].append(pr_auc)
                        epoch_stats["val_f1"].append(m["f1"])
                        epoch_stats["val_precision"].append(m["precision"])
                        epoch_stats["val_recall"].append(m["recall"])

                        last_epoch_stats = defaultdict()
                        for key in epoch_stats.keys():
                            last_epoch_stats[key] = epoch_stats[key][-1]


                        log(f"epoch {epoch} stats: {dict(last_epoch_stats)}")

                        scheduler.step(pr_auc)

                        if pr_auc > best_pr_auc:
                            best_pr_auc      = pr_auc
                            best_epoch       = epoch
                            best_model_state   = {k: v.cpu() for k, v in model.state_dict().items()}
                            best_encoder_state = {k: v.cpu() for k, v in encoder.state_dict().items()}

                # Restore best checkpoint
                model.load_state_dict(best_model_state)
                encoder.load_state_dict(best_encoder_state)

                _, val_probs, val_labels = self._eval_step(model, encoder, data, val_mask, pos_weight)
                threshold = self._find_best_threshold(val_labels.numpy(), val_probs.numpy())

                _, test_probs, test_labels = self._eval_step(model, encoder, data, test_mask, pos_weight)
                test_m = self._eval_with_threshold(test_probs, test_labels, threshold)

                fold_cms.append(test_m["confusion_matrix"])
                fold_mcen_aux.append(test_m["mcen"])
                fold_results["acc"].append(test_m["acc"])
                fold_results["recall"].append(test_m["recall"])
                fold_results["precision"].append(test_m["precision"])
                fold_results["f1_score"].append(test_m["f1"])

                log(f"Fold {fold_idx} | BEST EPOCH={best_epoch} | PR-AUC(val)={best_pr_auc:.4f} | "
                    f"Threshold={threshold:.3f} | Test F1={test_m['f1']:.4f} | "
                    f"Precision={test_m['precision']:.4f} | Recall={test_m['recall']:.4f} | "
                    f"MCEN(aux)={test_m['mcen']:.4f}")

            for k in fold_results:
                results[k].append(np.mean(fold_results[k]))

            global_cm = sum(fold_cms)
            log(f"MCEN global = {self._compute_mcen(global_cm):.4f}")
            log(f"MCEN per fold: mean={np.mean(fold_mcen_aux):.4f}, std={np.std(fold_mcen_aux):.4f}")

        log(f"cv training results = {dict(results)}")
        return results

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _build(self, rep, data):
        torch.manual_seed(rep)
        encoder = InputEncoder(data, proj_dim=32)
        model   = SimpleHGN(
            edge_dim=8,
            num_etypes=len(data.etypes),
            in_dim=[32],
            hidden_dim=16,
            num_classes=2,
            num_layers=2,
            heads=[4, 4, 1],
            feat_drop=0.1,
            negative_slope=0.2,
            residual=True,
            beta=0.1,
            ntypes=data.ntypes,
        )
        return model, encoder

    def _train_step(self, model, encoder, optimizer, data, mask, pos_weight):
        model.train(); encoder.train()
        optimizer.zero_grad()

        h_dict = encoder(data)
        logits = model(data, h_dict)['transaction'][mask]
        labels = data.nodes['transaction'].data['label'][mask].long()

        # Użycie Focal Loss
        weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=logits.device)
        focal_criterion = FocalLoss(weight=weight, gamma=2.0)
        loss = focal_criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(encoder.parameters()), max_norm=1.0
        )
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _eval_step(self, model, encoder, data, mask, pos_weight):
        model.eval(); encoder.eval()

        h_dict = encoder(data)
        logits = model(data, h_dict)['transaction'][mask]
        labels = data.nodes['transaction'].data['label'][mask].long()

        # Użycie Focal Loss
        weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=logits.device)
        focal_criterion = FocalLoss(weight=weight, gamma=2.0)
        loss = focal_criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        return loss.item(), probs.cpu(), labels.cpu()

    def _find_best_threshold(self, y_true, y_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f1  = 2 * precision * recall / (precision + recall + 1e-10)
        idx = int(np.argmax(f1[:-1]))   # thresholds ma len-1 elementów
        t   = float(thresholds[idx])
        log(f"--- Optymalizacja progu ---")
        log(f"Najlepszy próg: {t:.4f}")
        log(f"F1: {f1[idx]:.4f} | Precision: {precision[idx]:.4f} | Recall: {recall[idx]:.4f}")
        return t

    def _eval_with_threshold(self, probs, labels, threshold):
        preds  = (probs  >= threshold).int()
        labels = labels.int()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        eps  = 1e-12
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        cm   = np.array([[tn, fp], [fn, tp]])
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "acc": (tp + tn) / (tp + tn + fp + fn + eps),
                "precision": prec, "recall": rec, "f1": f1,
                "confusion_matrix": cm, "mcen": self._compute_mcen(cm)}

    def _calc_pos_weight(self, labels):
        n_neg = (labels == 0).sum().item()
        n_pos = (labels == 1).sum().item()
        return self.loss_mul * n_neg / max(n_pos, 1)

    def _compute_mcen(self, cm, eps=1e-12):
        cm = np.array(cm, dtype=float)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        d0 = (TN + FP) + (TN + FN)
        d1 = (FN + TP) + (FP + TP)
        c0 = -(FP / (d0 + eps)) * np.log(FP / (d0 + eps) + eps) if FP > 0 else 0.0
        c1 = -(FN / (d1 + eps)) * np.log(FN / (d1 + eps) + eps) if FN > 0 else 0.0
        return 0.5 * (c0 + c1)

        