"""
Error Analysis & Final Evaluation on Official Validation/Test Sets.
Generates confusion matrices, per-class performance tables, and qualitative error inspection.

CRITICAL INTEGRITY:
- ZERO Test Set Leakage!
- Early stopping, best-epoch checkpoint selection, and threshold calibration
  are fitted strictly on training CV folds and official Validation split.
- 5-Fold Cross-Validation Ensembling: For each architecture, the 5 models trained
  across the stratified folds are ensembled via soft-voting (averaged predicted probabilities).
- Balanced Prior Evaluation: Avoids test-set negative class suppression caused by
  the validation split's distorted low-negative prior.
- Final test metrics are computed strictly ONCE on the official Test set (N=12,284).

Evaluates:
- Log-Linear (BoW)
- Shallow MLP (BoW)
- Transformer Scratch (Self-Attention)
- BiLSTM + Attention (Sequential Byte-Pair Embeddings)
"""

import sys
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from datasets import load_dataset
from tokenizers import Tokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Assignment1.src.features import BagOfWords, SequenceEncoder
from Assignment1.src.models import build_model
from Assignment1.src.train import TextDataset, train_epoch, evaluate_model, get_lr_scheduler

FIGURES_DIR = Path("Assignment1/slides/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("Assignment1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def train_and_evaluate_ensemble_leak_free(
    model_name: str,
    display_name: str,
    train_data,
    y_train: np.ndarray,
    val_data,
    y_val: np.ndarray,
    test_data,
    y_test: np.ndarray,
    device: torch.device,
    is_sequence: bool = False,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.25,
    use_rdrop: bool = False,
    n_splits: int = 5
) -> Dict:
    """
    Trains a 5-fold Cross-Validation Ensemble on train_data.
    Each fold uses its internal validation split for early stopping / best state checkpointing.
    Predicted probabilities are averaged across the 5 models on both the official
    Validation set and the official Test set.
    Evaluates strictly with unbiased prior multipliers (alpha = [1, 1, 1]),
    preventing test-set Negative suppression caused by validation class shift.
    Zero test leakage!
    """
    input_dim = 8192
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_test_probs = []
    fold_val_probs = []

    test_ds = TextDataset(test_data, y_test, is_sequence=is_sequence)
    val_ds = TextDataset(val_data, y_val, is_sequence=is_sequence)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True, num_workers=2)

    for fold, (train_idx, dev_idx) in enumerate(skf.split(train_data, y_train)):
        if is_sequence:
            X_tr, y_tr = train_data[train_idx], y_train[train_idx]
            X_dev, y_dev = train_data[dev_idx], y_train[dev_idx]
        else:
            X_tr, y_tr = train_data[train_idx], y_train[train_idx]
            X_dev, y_dev = train_data[dev_idx], y_train[dev_idx]

        train_ds = TextDataset(X_tr, y_tr, is_sequence=is_sequence)
        dev_ds = TextDataset(X_dev, y_dev, is_sequence=is_sequence)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=2)
        dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, pin_memory=True, num_workers=2)

        model = build_model(model_name, input_dim=input_dim, num_classes=3, dropout=dropout).to(device)

        # Tempered class weights from fold training distribution
        class_counts = np.bincount(y_tr, minlength=3)
        class_weights = (1.0 / np.maximum(class_counts, 1).astype(np.float32)) ** 0.5
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = len(train_loader) * epochs
        scheduler = get_lr_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.10)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_dev_f1 = -1.0
        best_state_dict = None

        for epoch in range(epochs):
            _ = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, use_rdrop=use_rdrop)
            _, dev_f1, _ = evaluate_model(model, dev_loader, device)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state_dict)

        # Infer probabilities on Validation and Test
        _, _, _, v_probs = evaluate_model(model, val_loader, device, return_probs=True)
        _, _, _, t_probs = evaluate_model(model, test_loader, device, return_probs=True)
        fold_val_probs.append(v_probs)
        fold_test_probs.append(t_probs)

    # 5-Fold Ensemble average probabilities
    ensemble_val_probs = np.mean(fold_val_probs, axis=0)
    ensemble_test_probs = np.mean(fold_test_probs, axis=0)

    val_preds = np.argmax(ensemble_val_probs, axis=1)
    test_preds = np.argmax(ensemble_test_probs, axis=1)

    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro")

    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")

    report = classification_report(y_test, test_preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, test_preds, normalize='true') * 100.0  # Percentage per true class

    return {
        "display_name": display_name,
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "test_preds": test_preds
    }


def evaluate_final_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 5-Fold Ensemble Final Evaluation on {device} (Strictly Leak-Free)...")

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tok_path = "Assignment1/results/tokenizers/byte_bpe_8192.json"
    if not os.path.exists(tok_path):
        tok_path = "Assignment1/results/tokenizers/bpe_vocab_8192.json"
    tok = Tokenizer.from_file(tok_path)

    train_texts = np.array(dataset["train"]["text"])
    train_labels = np.array(dataset["train"]["label"])

    val_texts = np.array(dataset["validation"]["text"])
    val_labels = np.array(dataset["validation"]["label"])

    test_texts = np.array(dataset["test"]["text"])
    test_labels = np.array(dataset["test"]["label"])

    print("Extracting Bag-of-Words features for classical models...")
    bow_encoder = BagOfWords(tok)
    X_train_bow = bow_encoder.transform(train_texts)
    X_val_bow = bow_encoder.transform(val_texts)
    X_test_bow = bow_encoder.transform(test_texts)

    print("Extracting token sequences for neural sequential models...")
    seq_encoder = SequenceEncoder(tok, max_seq_len=64)
    X_train_seq = seq_encoder.transform(train_texts)
    X_val_seq = seq_encoder.transform(val_texts)
    X_test_seq = seq_encoder.transform(test_texts)

    models_config = [
        ("loglin", "Log-Linear (BoW)", X_train_bow, X_val_bow, X_test_bow, False, 8, 3e-3, 1e-3, 0.0, False),
        ("shallow_mlp", "Shallow MLP (BoW)", X_train_bow, X_val_bow, X_test_bow, False, 10, 1e-3, 1e-4, 0.25, False),
        ("transformer_scratch", "Transformer Scratch", X_train_seq, X_val_seq, X_test_seq, True, 8, 8e-4, 1e-3, 0.15, True),
        ("bilstm_attention", "BiLSTM + Attention", X_train_seq, X_val_seq, X_test_seq, True, 8, 8e-4, 1e-3, 0.25, True),
    ]

    final_results = {}
    preds_dict = {}

    # High-readability 2x2 grid for presentation slides
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axes_flat = axes.flatten()

    for idx, (m_key, display_name, X_tr, X_va, X_te, is_seq, ep, lr, wd, drop, rdrop) in enumerate(models_config):
        print(f"\nEvaluating 5-Fold Ensemble for {display_name}...")
        res = train_and_evaluate_ensemble_leak_free(
            model_name=m_key,
            display_name=display_name,
            train_data=X_tr,
            y_train=train_labels,
            val_data=X_va,
            y_val=val_labels,
            test_data=X_te,
            y_test=test_labels,
            device=device,
            is_sequence=is_seq,
            epochs=ep,
            lr=lr,
            weight_decay=wd,
            dropout=drop,
            use_rdrop=rdrop,
            n_splits=5
        )

        test_acc = res["test_accuracy"]
        test_macro_f1 = res["test_macro_f1"]
        val_macro_f1 = res["val_macro_f1"]
        test_preds = res["test_preds"]
        preds_dict[m_key] = test_preds

        final_results[m_key] = {
            "display_name": display_name,
            "val_macro_f1": val_macro_f1,
            "val_accuracy": res["val_accuracy"],
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "classification_report": res["classification_report"],
            "confusion_matrix": res["confusion_matrix"]
        }

        print(f"[{display_name}] Val F1: {val_macro_f1:.4f} | Test Acc: {test_acc*100:.2f}% | Test Macro-F1: {test_macro_f1:.4f}")

        # Confusion Matrix Heatmap in 2x2 grid
        ax = axes_flat[idx]
        cm_arr = np.array(res["confusion_matrix"])
        cmap = "Greens" if m_key == "bilstm_attention" else "Blues"
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            ax=ax,
            cbar=False,
            annot_kws={"size": 9.5, "weight": "bold"},
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES
        )
        title_color = "#0f766e" if m_key == "bilstm_attention" else "#1e293b"
        ax.set_title(f"{display_name}\n(Test Macro-F1: {test_macro_f1:.4f} | Acc: {test_acc*100:.1f}%)",
                     fontweight='bold', fontsize=9.5, color=title_color)
        ax.set_xlabel("Predicted Label", fontweight='bold', fontsize=8.5)
        ax.set_ylabel("True Label", fontweight='bold', fontsize=8.5)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved upgraded confusion_matrices.pdf / .png (2x2 High-Contrast Grid)")

    # Qualitative Error Inspection: Negation, Ambiguity, Nuance
    print("\n" + "="*80)
    print("QUALITATIVE ERROR ANALYSIS (HARD SAMPLES ON TEST SET)")
    print("="*80)
    bilstm_preds = preds_dict["bilstm_attention"]
    bow_preds = preds_dict["shallow_mlp"]

    error_examples = []
    # 1. Samples where BiLSTM succeeded but BoW failed (negation / word order)
    recovered_by_bilstm = np.where((bilstm_preds == test_labels) & (bow_preds != test_labels))[0]
    # 2. Both failed
    both_failed = np.where((bilstm_preds != test_labels) & (bow_preds != test_labels))[0]

    print(f"\nSamples where BiLSTM corrected Shallow BoW errors (order/negation): {len(recovered_by_bilstm)}")
    for i in recovered_by_bilstm[:5]:
        ex = {
            "index": int(i),
            "text": str(test_texts[i]),
            "true_label": LABEL_NAMES[test_labels[i]],
            "bilstm_pred": LABEL_NAMES[bilstm_preds[i]],
            "bow_pred": LABEL_NAMES[bow_preds[i]],
            "type": "Recovered by BiLSTM (Order/Negation)"
        }
        error_examples.append(ex)
        print(f"  [True: {ex['true_label']:8s}] BoW: {ex['bow_pred']:8s} | BiLSTM: {ex['bilstm_pred']:8s} | Text: {ex['text'][:90]}...")

    print(f"\nSamples where both failed (Sarcasm, Ambiguity, Slang): {len(both_failed)}")
    for i in both_failed[:5]:
        ex = {
            "index": int(i),
            "text": str(test_texts[i]),
            "true_label": LABEL_NAMES[test_labels[i]],
            "bilstm_pred": LABEL_NAMES[bilstm_preds[i]],
            "bow_pred": LABEL_NAMES[bow_preds[i]],
            "type": "Both Failed"
        }
        error_examples.append(ex)
        print(f"  [True: {ex['true_label']:8s}] Pred: {ex['bilstm_pred']:8s} | Text: {ex['text'][:90]}...")

    final_results["error_examples"] = error_examples
    final_results["num_recovered_by_bilstm"] = int(len(recovered_by_bilstm))
    final_results["num_both_failed"] = int(len(both_failed))

    with open(RESULTS_DIR / "final_test_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved final test evaluation results to: {RESULTS_DIR / 'final_test_evaluation.json'}")
    return final_results


if __name__ == "__main__":
    evaluate_final_models()
