"""
Tests for dataset/loader.py

Run from the project root:
    pytest tests/test_loader.py -v

All tests use max_samples=50 to avoid downloading/processing the full dataset
during testing. The first run will download and cache Tiny ImageNet (~161 MB).
"""

import numpy as np
import pytest

from dataset.loader import DataConfig, get_raw, get_numpy, get_dataloaders

# Small sample count so tests run quickly
N = 50


# ---------------------------------------------------------------------------
# get_raw()
# ---------------------------------------------------------------------------

class TestGetRaw:
    def test_train_split_loads(self):
        ds = get_raw("train")
        assert len(ds) > 0

    def test_valid_split_loads(self):
        ds = get_raw("valid")
        assert len(ds) > 0

    def test_invalid_split_raises(self):
        with pytest.raises(AssertionError):
            get_raw("test")

    def test_sample_has_image_and_label(self):
        ds = get_raw("train")
        sample = ds[0]
        assert "image" in sample
        assert "label" in sample

    def test_image_size(self):
        ds = get_raw("train")
        img = ds[0]["image"]
        assert img.size == (64, 64), f"Expected (64, 64), got {img.size}"
        assert img.mode == "RGB"

    def test_label_range(self):
        ds = get_raw("train")
        label = ds[0]["label"]
        assert 0 <= label <= 199, f"Label {label} out of range [0, 199]"


# ---------------------------------------------------------------------------
# get_numpy()
# ---------------------------------------------------------------------------

class TestGetNumpy:
    @pytest.fixture(scope="class")
    def arrays(self):
        cfg = DataConfig(max_samples=N)
        return get_numpy(cfg)

    def test_returns_four_arrays(self, arrays):
        assert len(arrays) == 4

    def test_X_train_shape(self, arrays):
        X_train, _, _, _ = arrays
        assert X_train.shape == (N, 12288), f"Got {X_train.shape}"

    def test_X_val_shape(self, arrays):
        _, _, X_val, _ = arrays
        assert X_val.shape == (N, 12288), f"Got {X_val.shape}"

    def test_y_train_shape(self, arrays):
        _, y_train, _, _ = arrays
        assert y_train.shape == (N,)

    def test_y_val_shape(self, arrays):
        _, _, _, y_val = arrays
        assert y_val.shape == (N,)

    def test_X_dtype(self, arrays):
        X_train, _, _, _ = arrays
        assert X_train.dtype == np.float32

    def test_y_dtype(self, arrays):
        _, y_train, _, _ = arrays
        assert y_train.dtype == np.int64

    def test_normalized_range(self, arrays):
        X_train, _, X_val, _ = arrays
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_val.min() >= 0.0
        assert X_val.max() <= 1.0

    def test_unnormalized_range(self):
        cfg = DataConfig(max_samples=N, normalize=False)
        X_train, _, _, _ = get_numpy(cfg)
        assert X_train.max() > 1.0, "Expected raw [0, 255] pixel values"

    def test_label_range(self, arrays):
        _, y_train, _, y_val = arrays
        assert y_train.min() >= 0 and y_train.max() <= 199
        assert y_val.min() >= 0 and y_val.max() <= 199

    def test_max_samples_respected(self):
        cfg = DataConfig(max_samples=10)
        X_train, y_train, X_val, y_val = get_numpy(cfg)
        assert X_train.shape[0] == 10
        assert X_val.shape[0] == 10


# ---------------------------------------------------------------------------
# get_dataloaders()
# ---------------------------------------------------------------------------

class TestGetDataloaders:
    @pytest.fixture(scope="class")
    def loaders(self):
        cfg = DataConfig(batch_size=16, max_samples=N, num_workers=0)
        return get_dataloaders(cfg)

    def test_returns_two_loaders(self, loaders):
        assert len(loaders) == 2

    def test_train_batch_image_shape(self, loaders):
        train_loader, _ = loaders
        images, labels = next(iter(train_loader))
        assert images.shape == (16, 3, 64, 64), f"Got {images.shape}"

    def test_val_batch_image_shape(self, loaders):
        _, val_loader = loaders
        images, labels = next(iter(val_loader))
        assert images.shape[1:] == (3, 64, 64)

    def test_batch_image_dtype(self, loaders):
        import torch
        train_loader, _ = loaders
        images, _ = next(iter(train_loader))
        assert images.dtype == torch.float32

    def test_batch_image_range(self, loaders):
        train_loader, _ = loaders
        images, _ = next(iter(train_loader))
        assert images.min().item() >= 0.0
        assert images.max().item() <= 1.0

    def test_batch_label_shape(self, loaders):
        train_loader, _ = loaders
        _, labels = next(iter(train_loader))
        assert labels.shape == (16,)

    def test_batch_label_range(self, loaders):
        train_loader, _ = loaders
        _, labels = next(iter(train_loader))
        assert labels.min().item() >= 0
        assert labels.max().item() <= 199

    def test_num_batches(self, loaders):
        train_loader, val_loader = loaders
        # N=50 samples, batch_size=16 -> ceil(50/16) = 4 batches
        assert len(train_loader) == 4
        assert len(val_loader) == 4

    def test_max_samples_respected(self):
        cfg = DataConfig(batch_size=8, max_samples=8, num_workers=0)
        train_loader, val_loader = get_dataloaders(cfg)
        assert len(train_loader) == 1
        assert len(val_loader) == 1
