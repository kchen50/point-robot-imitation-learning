import argparse

import torch

from policy.policy import STATE_DIM, ACTION_DIM
from policy.actor_critic_policy import Actor as Policy
from training.bc_mle_gaussian import train_BC_policy



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a behavior cloning policy for the point robot."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        # default="data/2026-02-21_23-19-38-trajectories-500",
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy = Policy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    _, _, _, best_policy = train_BC_policy(
        policy,
        dataset_root=args.data_path,
        num_epochs=args.epochs,
        batch_size_train=args.batch_size,
        lr=args.lr,
    )

    # train_BC_policy already restores best weights; reload for clarity
    if best_policy is not None:
        policy.load_state_dict(best_policy)

    # Actor has no static save; save state_dict for later load_state_dict
    torch.save(policy.state_dict(), "./bc_mle_policy.pth")
