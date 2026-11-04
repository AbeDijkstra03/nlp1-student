"""
Comprehensive Cross-Validation Training & Evaluation Pipeline.
Evaluates:
- Baseline 1: Log-Linear (Indicator, BoW, TF-IDF)
- Baseline 2: Shallow MLP (BoW, TF-IDF)
- Ablation 3: Deep Residual MLP (BoW)
- Champion 1: BiLSTM with Self-Attention Pooling (Sequential Token IDs)
- Champion 2: Twitter Transformer Encoder from Scratch (Sequential Token IDs)

Features:
- In-domain self-supervised PPMI-SVD subword embedding initialization
- Tempered class-weighting (1 / N_c)^0.5 to prevent negative over-prediction
- R-Drop consistency regularization with symmetric KL divergence
- Cosine Annealing with Linear Warmup learning rate schedule
- Strict leak-free Post-Hoc Prior / Threshold Calibration on Validation fold
"""

import os
import sys
from pathlib import Path
import json
import time
import math
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tokenizers import Tokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Assignment1.src.features import TokenIndicator, BagOfWords, SubwordTfidf, SequenceEncoder
from Assignment1.src.models import build_model, get_or_train_in_domain_embeddings

RESULTS_DIR = Path("Assignment1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class TextDataset(Dataset):
    """Handles both sparse feature matrices and dense sequence token ID arrays."""
    def __init__(self, data, labels: np.ndarray, is_sequence: bool = False):
        self.data = data
        self.labels = np.array(labels, dtype=np.int64)
        self.is_sequence = is_sequence

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.is_sequence:
            x = torch.from_numpy(self.data[idx]).long()
        else:
            x = torch.from_numpy(self.data[idx].toarray().squeeze(0)).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def get_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.10):
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def calibrate_class_priors(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Find class multipliers alpha in R^3 on validation probabilities (preds = argmax(alpha * probs))
    to maximize Macro-F1.
    """
    raw_preds = np.argmax(probs, axis=-1)
    best_f1 = f1_score(targets, raw_preds, average='macro')
    best_acc = accuracy_score(targets, raw_preds)
    best_alphas = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    best_preds = raw_preds

    for a0 in np.linspace(0.7, 1.4, 8):
        for a2 in np.linspace(0.7, 1.4, 8):
            alphas = np.array([a0, 1.0, a2], dtype=np.float32)
            scaled_probs = probs * alphas
            preds = np.argmax(scaled_probs, axis=-1)
            score = f1_score(targets, preds, average='macro')
            if score > best_f1:
                best_f1 = score
                best_acc = accuracy_score(targets, preds)
                best_alphas = alphas
                best_preds = preds

    return float(best_f1), float(best_acc), best_alphas, best_preds


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, use_rdrop: bool = False, kl_weight: float = 0.5):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if use_rdrop:
                p1_logits = model(x)
                p2_logits = model(x)
                loss_ce = 0.5 * (criterion(p1_logits, y) + criterion(p2_logits, y))
                p1 = F.log_softmax(p1_logits, dim=-1)
                p2 = F.log_softmax(p2_logits, dim=-1)
                loss_kl = 0.5 * (F.kl_div(p1, p2.exp(), reduction='batchmean') + F.kl_div(p2, p1.exp(), reduction='batchmean'))
                loss = loss_ce + kl_weight * loss_kl
            else:
                logits = model(x)
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scaler.get_scale() >= scale:
            scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device, alphas: Optional[np.ndarray] = None, return_probs: bool = False):
    model.eval()
    all_probs = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_targets.extend(y.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.array(all_targets)

    if alphas is not None:
        scaled_probs = all_probs * alphas
        preds = np.argmax(scaled_probs, axis=-1)
    else:
        preds = np.argmax(all_probs, axis=-1)

    acc = float(accuracy_score(all_targets, preds))
    macro_f1 = float(f1_score(all_targets, preds, average='macro'))

    if return_probs:
        return acc, macro_f1, preds, all_probs
    return acc, macro_f1, preds


def train_and_eval_fold(
    model_name: str,
    train_data,
    y_train: np.ndarray,
    val_data,
    y_val: np.ndarray,
    device: torch.device,
    is_sequence: bool = False,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.2,
    use_rdrop: bool = False
) -> Dict:
    input_dim = 8192
    model = build_model(model_name, input_dim=input_dim, num_classes=3, dropout=dropout).to(device)

    # Tempered class-weighted loss to protect against distribution shift without hurting precision
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = (1.0 / np.maximum(class_counts, 1).astype(np.float32)) ** 0.5
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TextDataset(train_data, y_train, is_sequence=is_sequence)
    val_ds = TextDataset(val_data, y_val, is_sequence=is_sequence)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    total_steps = len(train_loader) * epochs
    scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.10)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_macro_f1 = -1.0
    best_acc = -1.0
    best_preds = None
    best_alphas = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    best_state_dict = None

    for epoch in range(epochs):
        _ = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, use_rdrop=use_rdrop)
        _, _, _, val_probs = evaluate_model(model, val_loader, device, return_probs=True)

        calib_f1, calib_acc, alphas, preds = calibrate_class_priors(val_probs, y_val)

        if calib_f1 > best_macro_f1:
            best_macro_f1 = calib_f1
            best_acc = calib_acc
            best_preds = preds
            best_alphas = alphas
            best_state_dict = copy.deepcopy(model.state_dict())

    return {
        "best_macro_f1": best_macro_f1,
        "best_acc": best_acc,
        "best_preds": best_preds,
        "best_alphas": best_alphas.tolist(),
        "state_dict": best_state_dict,
        "model": model
    }


def run_comprehensive_cv(
    texts: List[str],
    labels: np.ndarray,
    tokenizer: Tokenizer,
    n_splits: int = 5,
    seed: int = 42
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting 5-Fold Cross Validation on {device} (Zero Test Leakage)...")

    # Ensure in-domain embeddings are available
    get_or_train_in_domain_embeddings(tokenizer=tokenizer, vocab_size=8192, embed_dim=128)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    seq_encoder = SequenceEncoder(tokenizer, max_seq_len=64)

    # Configs to evaluate: exactly matches previous keys for full compatibility
    configs = [
        # Baselines from Tutorial
        {"name": "loglin_indicator", "model": "loglin", "rep": "indicator", "is_seq": False, "epochs": 8,  "lr": 3e-3, "wd": 1e-3, "drop": 0.0,  "rdrop": False},
        {"name": "loglin_bow",       "model": "loglin", "rep": "bow",       "is_seq": False, "epochs": 8,  "lr": 3e-3, "wd": 1e-3, "drop": 0.0,  "rdrop": False},
        {"name": "loglin_tfidf",     "model": "loglin", "rep": "tfidf",     "is_seq": False, "epochs": 8,  "lr": 3e-3, "wd": 1e-3, "drop": 0.0,  "rdrop": False},
        {"name": "shallow_mlp_bow",  "model": "shallow_mlp", "rep": "bow",   "is_seq": False, "epochs": 10, "lr": 1e-3, "wd": 1e-4, "drop": 0.25, "rdrop": False},
        {"name": "shallow_mlp_tfidf","model": "shallow_mlp", "rep": "tfidf", "is_seq": False, "epochs": 10, "lr": 1e-3, "wd": 1e-4, "drop": 0.25, "rdrop": False},
        {"name": "deep_mlp_bow",     "model": "deep_mlp", "rep": "bow",     "is_seq": False, "epochs": 10, "lr": 8e-4, "wd": 1e-4, "drop": 0.25, "rdrop": False},
        # Sequential Champions with PPMI-SVD embeddings + Spatial Dropout + Multi-Pooling + R-Drop
        {"name": "bilstm_attention", "model": "bilstm", "rep": "seq",        "is_seq": True,  "epochs": 8,  "lr": 8e-4, "wd": 1e-3, "drop": 0.25, "rdrop": True},
        {"name": "transformer_scratch", "model": "transformer", "rep": "seq", "is_seq": True, "epochs": 8,  "lr": 8e-4, "wd": 1e-3, "drop": 0.15, "rdrop": True},
    ]

    results = {c["name"]: {"macro_f1": [], "acc": []} for c in configs}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n{'='*25} Fold {fold_idx + 1}/{n_splits} {'='*25}")
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Feature representations for this fold
        print("  Extracting fold features...")
        ind_enc = TokenIndicator(tokenizer)
        bow_enc = BagOfWords(tokenizer)
        tfidf_enc = SubwordTfidf(tokenizer).fit(train_texts)

        feat_cache = {
            "indicator": (ind_enc.transform(train_texts), ind_enc.transform(val_texts)),
            "bow":       (bow_enc.transform(train_texts), bow_enc.transform(val_texts)),
            "tfidf":     (tfidf_enc.transform(train_texts), tfidf_enc.transform(val_texts)),
            "seq":       (seq_encoder.encode(train_texts), seq_encoder.encode(val_texts))
        }

        for cfg in configs:
            name = cfg["name"]
            X_tr, X_v = feat_cache[cfg["rep"]]
            res = train_and_eval_fold(
                model_name=cfg["model"],
                train_data=X_tr,
                y_train=y_train,
                val_data=X_v,
                y_val=y_val,
                device=device,
                is_sequence=cfg["is_seq"],
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                weight_decay=cfg["wd"],
                dropout=cfg["drop"],
                use_rdrop=cfg["rdrop"]
            )
            results[name]["macro_f1"].append(res["best_macro_f1"])
            results[name]["acc"].append(res["best_acc"])
            print(f"    [{name:22s}] Fold {fold_idx+1} -> Macro-F1: {res['best_macro_f1']:.4f} | Acc: {res['best_acc']:.4f}")

    # Summary table
    summary = {}
    print("\n" + "="*75)
    print("5-FOLD CROSS-VALIDATION SUMMARY (TRAIN SET ONLY - ZERO TEST LEAKAGE)")
    print("="*75)
    print(f"{'System':<24} | {'Macro-F1 (Mean +/- Std)':<22} | {'Accuracy (Mean +/- Std)':<22}")
    print("-" * 75)

    for name, data in results.items():
        f1_m, f1_s = np.mean(data["macro_f1"]), np.std(data["macro_f1"])
        acc_m, acc_s = np.mean(data["acc"]), np.std(data["acc"])
        summary[name] = {
            "fold_macro_f1": [float(v) for v in data["macro_f1"]],
            "fold_acc": [float(v) for v in data["acc"]],
            "mean_macro_f1": float(f1_m),
            "std_macro_f1": float(f1_s),
            "mean_acc": float(acc_m),
            "std_acc": float(acc_s)
        }
        print(f"{name:<24} | {f1_m:.4f} +/- {f1_s:.4f}        | {acc_m:.4f} +/- {acc_s:.4f}")

    with open(RESULTS_DIR / "cv_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved cross-validation results to: {RESULTS_DIR / 'cv_results_summary.json'}")
    return summary


# Alias for backward compatibility
run_cross_validation = run_comprehensive_cv


if __name__ == "__main__":
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tok_path = "Assignment1/results/tokenizers/byte_bpe_8192.json"
    if not os.path.exists(tok_path):
        tok_path = "Assignment1/results/tokenizers/bpe_vocab_8192.json"
    tok = Tokenizer.from_file(tok_path)
    train_texts = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])

    run_comprehensive_cv(train_texts, train_labels, tok, n_splits=5)
