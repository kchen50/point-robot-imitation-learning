import torch
import torch.nn as nn

import copy

from policy.policy import Policy
from utils.pytorch import make_step_dataloaders
try:
    from tqdm import tqdm as _tqdm
except Exception:
    class _tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=True):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1): pass
        def set_postfix_str(self, s): pass
        def close(self): pass

def train_BC_policy(
    policy: Policy,
    dataset_root: str,
    num_epochs: int = 100,
    batch_size_train: int = 1024,
    lr: float = 1e-3,
    device: str | None = None,
    patience: int = 5,
    min_delta: float = 0.0,
    verbose: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)

    # Build dataloaders
    train_loader, val_loader, _ = make_step_dataloaders(
        root=dataset_root,
        batch_size_train=batch_size_train,
        ratios=(0.8, 0.1, 0.1),
        seed=42,
        num_workers_train=2,
        pin_memory=True,
        persistent_workers=True,
        full_batch_val=True,
        full_batch_test=True,
    )

    # Create a loss function
    loss_fn = nn.MSELoss()

    # Create an optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Train the policy using behavior cloning
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_val_epoch = -1
    best_policy = None
    bad_epochs = 0
    
    # TODO: Implement the behavior cloning algorithm!
        # Data: train_loader, val_loader
        # Model: policy
        # Loss: loss_fn
        # Optimizer: optimizer
        # Metrics: train_losses, val_losses

    # Restore best policy if we tracked one
    if best_policy is not None:
        policy.load_state_dict(best_policy)
    return train_losses, val_losses, best_val_epoch, best_policy

if __name__ == "__main__":
    policy = Policy()
    # data_path = "data/2026-02-21_21-20-06-trajectories-2500"
    data_path = "data/2026-02-21_23-19-38-trajectories-500"
    _, _, _, best_policy = train_BC_policy(policy, data_path)

    policy.load_state_dict(best_policy)

    Policy.save(policy, "./policy.pth")