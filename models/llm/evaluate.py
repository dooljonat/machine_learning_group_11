# LLM (Decoder-only) evaluation - Jonathan Dooley
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, TinyImageNetDataset, CHECKPOINT_DIR

K_FOLDS = 5
BATCH_SIZE = 128
MODEL_SIZE = "base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model(size=MODEL_SIZE):
    model = build_model(size).to(DEVICE)
    path = os.path.join(CHECKPOINT_DIR, f"{size}_best.pth")
    cp = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    print(f"Loaded {size} checkpoint — epoch={cp['epoch']}, val_acc={cp['val_acc']:.2f}%")
    return model


def predict(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(DEVICE)
            preds = model(patches).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def print_metrics(y_true, y_pred, label=""):
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if label:
        print(f"\n--- {label} ---")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return acc, prec, rec, f1


def run_kfold(ds_train):
    print(f"\n=== {K_FOLDS}-Fold Cross Validation ===")
    labels = np.array([ds_train.data[i]["label"] for i in range(len(ds_train))])
    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(ds_train))

    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, labels), 1):
        print(f"\n--- Fold {fold}/{K_FOLDS} ---")
        val_loader = DataLoader(
            Subset(ds_train, val_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        )
        model = load_best_model()
        y_true, y_pred = predict(model, val_loader)
        acc, prec, rec, f1 = print_metrics(y_true, y_pred)
        fold_accs.append(acc)
        fold_precs.append(prec)
        fold_recs.append(rec)
        fold_f1s.append(f1)

    print("\n--- Cross-Validation Summary ---")
    print(f"Accuracies:  {[f'{a:.2f}' for a in fold_accs]}")
    print(f"Avg Accuracy:  {np.mean(fold_accs):.2f}%")
    print(f"Avg Precision: {np.mean(fold_precs):.4f}")
    print(f"Avg Recall:    {np.mean(fold_recs):.4f}")
    print(f"Avg F1 Score:  {np.mean(fold_f1s):.4f}")


def run_final_eval(ds_val):
    print("\n=== Final Test Set Evaluation ===")
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = load_best_model()
    y_true, y_pred = predict(model, val_loader)
    print_metrics(y_true, y_pred, label="Final Test Set Metrics")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Confusion Matrix — Decoder-Only LLM ({MODEL_SIZE})")
    out = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out)
    print(f"\nSaved confusion matrix to {out}")


if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_dataset("zh-plus/tiny-imagenet")
    ds_train = TinyImageNetDataset(ds["train"], augment=False)
    ds_val = TinyImageNetDataset(ds["valid"], augment=False)

    run_kfold(ds_train)
    run_final_eval(ds_val)
