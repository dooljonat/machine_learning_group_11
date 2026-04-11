# Kernel SVM model - Jonathan Dooley

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from dataset.loader import DataConfig, get_numpy, get_kfold_numpy

RESULTS_DIR = Path(__file__).parent / "results"


def build_pipeline(C=10.0, gamma='scale', n_components=500):
    """
    Build and return a sklearn Pipeline:
    StandardScaler -> PCA -> RBF SVC.

    StandardScaler standardizes features by removing the mean and scaling to unit variance
    PCA is for dimensionality reduction and finding the most 'important' features for the classification task
    SVC is the kernel SVM using the RBF kernel and specified values of C and gamma
    """

    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=n_components, whiten=True, random_state=42)),
        ('svc',    SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape='ovo')),
    ])


def tune(config: DataConfig):
    """
    K-fold CV over a hyperparameter grid. Returns best (C, gamma) params.

    Uses get_kfold_numpy() so the HuggingFace 'valid' split stays held-out.
    """

    # Hyperparameters to search through
    param_grid = [
        (C, gamma)
        for C     in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        for gamma in [0.001, 0.01, 0.1, 1.0, 'scale', 'auto']
    ]

    # Search through possible hyperparameter combinations
    print(f"Tuning over {len(param_grid)} hyperparameter configs, {config.n_splits} folds each...")
    results = {}
    for i, (C, gamma) in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] C={C}, gamma={gamma}")
        fold_accs = []

        # Train SVM on current fold and get accuracies
        for fold, (X_tr, y_tr, X_v, y_v) in enumerate(get_kfold_numpy(config), 1):
            print(f"  fold {fold}/{config.n_splits}: fitting {X_tr.shape[0]} samples...", end=' ', flush=True)
            pipe = build_pipeline(C=C, gamma=gamma)
            pipe.fit(X_tr, y_tr)
            acc = (pipe.predict(X_v) == y_v).mean()
            fold_accs.append(acc)
            print(f"acc={acc:.4f}")

        # Get results for fold
        mean_acc = np.mean(fold_accs)
        results[(C, gamma)] = mean_acc
        print(f"  -> mean cv_acc={mean_acc:.4f}")

    # Return best hyperparameters for Kernel SVM based on k-fold cross validation
    best = max(results, key=results.get)
    print(f"\nBest: C={best[0]} gamma={best[1]}  acc={results[best]:.4f}")
    return {'C': best[0], 'gamma': best[1]}


def train_and_eval(best_params: dict, config: DataConfig):
    """
    Train on full (balanced) train split, evaluate on valid split.
    """
    
    print(f"\nLoading data for final eval (max_samples_per_class={config.max_samples_per_class})...")
    X_train, y_train, X_val, y_val = get_numpy(config)
    print(f"  train: {X_train.shape}, val: {X_val.shape}")

    print(f"Training final model with {X_train.shape[0]} samples...")
    pipe = build_pipeline(**best_params)
    pipe.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = pipe.predict(X_val)

    acc       = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_val, y_pred, average='macro', zero_division=0)
    cm        = confusion_matrix(y_val, y_pred)
    report    = classification_report(y_val, y_pred, zero_division=0)

    print(f"\n--- Final Metrics ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}  (macro)")
    print(f"  Recall    : {recall:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")
    print("\nPer-class report:")
    print(report)

    # --- Save results ---
    RESULTS_DIR.mkdir(exist_ok=True)

    # Metrics JSON
    metrics = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Per-class report text
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    return pipe, {**metrics, 'confusion_matrix': cm}


if __name__ == "__main__":
    # Train on full dataset using best hyperparameters from grid search
    # Best params found: C=100, gamma='auto'
    best_params = {'C': 100, 'gamma': 'auto'}

    # Train on full dataset (500 samples per class = 100,000 total samples)
    eval_cfg = DataConfig(max_samples_per_class=50)

    train_and_eval(best_params, eval_cfg)
