"""
Tiny ImageNet data loader.

Provides two output modes:
  - get_numpy()       -> (X_train, y_train, X_val, y_val) as flat float32 numpy arrays
                         For sklearn models: Decision Tree, Naive Bayes, SVM, Logistic Regression
  - get_kfold_numpy() -> generator of (X_tr, y_tr, X_v, y_v) over k stratified folds
                         For hyperparameter tuning; uses train split only, valid stays held-out
  - get_dataloaders() -> (train_loader, val_loader) as PyTorch DataLoaders
                         For neural net models: MLP, CNN, LSTM, Transformer, LLM
  - get_raw()         -> raw HuggingFace Dataset for a given split
                         For custom preprocessing in individual model folders

Usage example:
    from dataset.loader import DataConfig, get_numpy, get_dataloaders, get_kfold_numpy

    # Sklearn models
    config = DataConfig(max_samples=10000)
    X_train, y_train, X_val, y_val = get_numpy(config)

    # K-fold cross validation (for hyperparameter tuning)
    config = DataConfig(max_samples=10000, n_splits=5)
    for fold, (X_tr, y_tr, X_v, y_v) in enumerate(get_kfold_numpy(config)):
        ...

    # Neural net models
    config = DataConfig(batch_size=64)
    train_loader, val_loader = get_dataloaders(config)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # --- DataLoader mode ---
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 2

    # --- Both modes ---
    normalize: bool = True   # scale pixel values to [0.0, 1.0]
    max_samples: int = None  # cap dataset size per split (None = use all)
                             # recommended for slow models (SVM, Decision Tree)

    # --- K-fold mode ---
    n_splits: int = 5        # number of folds for get_kfold_numpy()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_hf(split: str):
    """Load a HuggingFace split, cached locally after first download."""
    from datasets import load_dataset as hf_load
    return hf_load("zh-plus/tiny-imagenet", split=split)


def _maybe_truncate(ds, max_samples):
    if max_samples is not None and max_samples < len(ds):
        return ds.select(range(max_samples))
    return ds

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_raw(split: str = "train"):
    """
    Return the raw HuggingFace Dataset for a given split.

    Args:
        split: "train" or "valid"

    Returns:
        datasets.Dataset with fields: image (PIL), label (int 0-199)
    """
    assert split in ("train", "valid"), f"split must be 'train' or 'valid', got '{split}'"
    return _load_hf(split)


def get_numpy(config: DataConfig = None):
    """
    Return flat numpy arrays suitable for sklearn models.

    Returns:
        X_train : float32 array, shape (N_train, 12288)
        y_train : int64  array, shape (N_train,)
        X_val   : float32 array, shape (N_val,   12288)
        y_val   : int64  array, shape (N_val,)

    Pixel values are in [0.0, 1.0] when config.normalize=True, else [0, 255].
    Feature engineering (PCA, HOG, etc.) should be done in each model's folder.
    """
    if config is None:
        config = DataConfig()

    def _split_to_arrays(split):
        ds = _maybe_truncate(_load_hf(split), config.max_samples)
        X = np.stack(
            [np.array(sample["image"], dtype=np.float32).flatten() for sample in ds]
        )
        y = np.array([sample["label"] for sample in ds], dtype=np.int64)
        if config.normalize:
            X /= 255.0
        return X, y

    X_train, y_train = _split_to_arrays("train")
    X_val,   y_val   = _split_to_arrays("valid")
    return X_train, y_train, X_val, y_val


def get_kfold_numpy(config: DataConfig = None):
    """
    Yield stratified k-fold splits over the training set.

    Each iteration yields:
        X_train_fold : float32 array, shape (N*(k-1)/k, 12288)
        y_train_fold : int64  array
        X_val_fold   : float32 array, shape (N/k, 12288)
        y_val_fold   : int64  array

    The HuggingFace 'valid' split is NOT used here — it remains the
    final held-out test set. Use this function for hyperparameter search only.
    """
    from sklearn.model_selection import StratifiedKFold

    if config is None:
        config = DataConfig()

    ds = _maybe_truncate(_load_hf("train"), config.max_samples)
    X = np.stack([np.array(s["image"], dtype=np.float32).flatten() for s in ds])
    y = np.array([s["label"] for s in ds], dtype=np.int64)
    if config.normalize:
        X /= 255.0

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def get_dataloaders(config: DataConfig = None):
    """
    Return PyTorch DataLoaders suitable for neural net models.

    Returns:
        train_loader : DataLoader, shuffled, yields (images, labels)
        val_loader   : DataLoader, unshuffled

    Image tensors have shape (batch_size, 3, 64, 64), dtype float32, values in [0, 1].
    Additional transforms (augmentation, normalization stats, resizing) should be
    applied in each model's folder by wrapping or subclassing the returned loader.
    """
    if config is None:
        config = DataConfig()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    to_tensor = transforms.ToTensor()  # PIL (H, W, C) uint8 -> (C, H, W) float32 in [0, 1]

    class TinyImageNetDataset(Dataset):
        def __init__(self, split):
            self.ds = _maybe_truncate(_load_hf(split), config.max_samples)

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            sample = self.ds[idx]
            return to_tensor(sample["image"]), sample["label"]

    train_loader = DataLoader(
        TinyImageNetDataset("train"),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        TinyImageNetDataset("valid"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick verification: python -m dataset.loader
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Numpy mode (max_samples=500) ===")
    cfg = DataConfig(max_samples=500)
    X_tr, y_tr, X_v, y_v = get_numpy(cfg)
    print(f"  X_train : {X_tr.shape}  dtype={X_tr.dtype}  range=[{X_tr.min():.2f}, {X_tr.max():.2f}]")
    print(f"  y_train : {y_tr.shape}  dtype={y_tr.dtype}")
    print(f"  X_val   : {X_v.shape}")
    print(f"  y_val   : {y_v.shape}")

    print("\n=== DataLoader mode (batch_size=32, max_samples=500) ===")
    cfg = DataConfig(batch_size=32, max_samples=500, num_workers=0)
    train_loader, val_loader = get_dataloaders(cfg)
    images, labels = next(iter(train_loader))
    print(f"  batch images : {tuple(images.shape)}  dtype={images.dtype}  range=[{images.min():.2f}, {images.max():.2f}]")
    print(f"  batch labels : {tuple(labels.shape)}  dtype={labels.dtype}")
    print(f"  train batches: {len(train_loader)}")
    print(f"  val batches  : {len(val_loader)}")

    print("\n=== K-Fold mode (n_splits=3, max_samples=300) ===")
    cfg = DataConfig(max_samples=300, n_splits=3)
    for fold, (X_tr, y_tr, X_v, y_v) in enumerate(get_kfold_numpy(cfg)):
        print(f"  fold {fold}: train={X_tr.shape}, val={X_v.shape}")
