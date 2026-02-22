import numpy as np
from typing import Tuple, Optional

try:
    import torch
    from torch.utils.data import DataLoader, Subset
except Exception as e:
    raise RuntimeError("PyTorch is required for utils.pytorch functions") from e

try:
    from dataset import PointRobotDatasetManager
except Exception as e:
    from utils.dataset import PointRobotDatasetManager

"""
Step dataset management
"""

def create_step_dataset(root: str, as_torch: bool = True):
    """
    Load a StepDataset (flat (s, a) samples) from a saved dataset root.
    """
    return PointRobotDatasetManager.get_step_dataset(root, as_torch=as_torch)


def _split_indices(n: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    """
    Produce train/val/test index arrays with a fixed RNG seed.
    """
    assert n >= 0
    r_train, r_val, r_test = ratios
    assert abs((r_train + r_val + r_test) - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(r_train * n))
    n_val = int(round(r_val * n))
    n_test = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def split_step_dataset(ds, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    """
    Split a StepDataset into Subsets (train, val, test).
    """
    n = len(ds)
    train_idx, val_idx, test_idx = _split_indices(n, ratios=ratios, seed=seed)
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)


def make_step_dataloaders(
    root: str,
    batch_size_train: int = 1024,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_workers_train: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    full_batch_val: bool = True,
    full_batch_test: bool = True,
):
    """
    Convenience: create (train_loader, val_loader, test_loader) for a StepDataset.
    - Train uses mini-batches with shuffling and workers.
    - Val/Test use full-batch if enabled; otherwise use the same batch size without shuffling.
    """
    ds = create_step_dataset(root, as_torch=True)
    train_ds, val_ds, test_ds = split_step_dataset(ds, ratios=ratios, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers_train,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers_train > 0 else False,
    )

    if len(val_ds) == 0:
        val_loader = None
    else:
        bs_val = len(val_ds) if full_batch_val else batch_size_train
        val_loader = DataLoader(
            val_ds,
            batch_size=bs_val,
            shuffle=False,
            drop_last=False,
            num_workers=0,  # deterministic and lightweight
            pin_memory=False,
        )

    if len(test_ds) == 0:
        test_loader = None
    else:
        bs_test = len(test_ds) if full_batch_test else batch_size_train
        test_loader = DataLoader(
            test_ds,
            batch_size=bs_test,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

    return train_loader, val_loader, test_loader

"""
Trajectory dataset management
"""

if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_step_dataloaders(
        root="data/2026-02-21_21-20-06-trajectories-2500",
        batch_size_train=1024,
        ratios=(0.8, 0.1, 0.1),
        seed=42,
        num_workers_train=4,
        pin_memory=True,
        persistent_workers=True,
    )