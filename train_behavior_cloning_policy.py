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
    for epoch in range(num_epochs):
        policy.train()
        epoch_train_losses = []
        it = 0
        for states, actions in _tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True) if verbose else train_loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            # Forward pass
            actions_pred = policy(states)
            train_loss = loss_fn(actions_pred, actions)
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()
            epoch_train_losses.append(train_loss.item())
            it += 1
        train_losses.append(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))

        # Validation pass
        if val_loader is not None:
            policy.eval()
            with torch.no_grad():
                epoch_val_losses = []
                for states_val, actions_val in val_loader:
                    states_val = states_val.to(device)
                    actions_val = actions_val.to(device)
                    val_actions_pred = policy(states_val)
                    val_loss = loss_fn(val_actions_pred, actions_val)
                    epoch_val_losses.append(val_loss.item())
                mean_val = sum(epoch_val_losses) / max(1, len(epoch_val_losses))
                val_losses.append(mean_val)
                if verbose:
                    print(f"Epoch {epoch+1}: train {train_losses[-1]:.6f} | val {mean_val:.6f}")
                # Early stopping check
                if mean_val + min_delta < best_val_loss:
                    best_val_loss = mean_val
                    best_val_epoch = epoch
                    best_policy = copy.deepcopy(policy.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1} (best epoch {best_val_epoch+1}, best val {best_val_loss:.6f})")
                    break
        else:
            val_losses.append(float('nan'))

    # Restore best policy if we tracked one
    if best_policy is not None:
        policy.load_state_dict(best_policy)
    return train_losses, val_losses, best_val_epoch, best_policy

if __name__ == "__main__":
    policy = Policy()
    # train_BC_policy(policy, "data/2026-02-21_23-19-38-trajectories-500")
    train_BC_policy(policy, "data/2026-02-21_21-20-06-trajectories-2500")