import torch
import torch.nn as nn

from utils.dataset import PointRobotDatasetManager

STATE_DIM = 4
ACTION_DIM = 2

class Policy(nn.Module):
    """
    A simple MLP-based policy that predicts a single action from a given state.
    """
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

        self.activation = nn.ReLU()

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x) # Output is in bounds [-1, 1]
        return x

def train_BC_policy(policy : Policy, dataset_manager : PointRobotDatasetManager, num_epochs : int = 100):
    # Create a data loader from the dataset manager
    data_loader = dataset_manager.create_data_loader()

    # Create a loss function
    loss_fn = nn.MSELoss()

    # Create an optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    # Train the policy using behavior cloning
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_val_epoch = -1
    best_policy = None
    for epoch in range(num_epochs):
        for batch in data_loader:
            states = batch['state']
            actions = batch['action']

            # Forward pass
            actions_pred = policy(states)
            train_loss = loss_fn(actions_pred, actions)

            train_losses.append(train_loss.item())

            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


        # Validation pass
        with torch.no_grad():
            val_actions_pred = policy(states_val)
            val_loss = loss_fn(val_actions_pred, actions_val)
            val_losses.append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_val_epoch = epoch
                best_policy = policy.state_dict()

    return train_losses, val_losses

def evaluate_policy(policy : Policy, dataset_manager : PointRobotDatasetManager):
    pass

def save_policy(policy : Policy, path : str):
    torch.save(policy.state_dict(), path)

def load_policy(path : str) -> Policy:
    policy = Policy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    policy.load_state_dict(torch.load(path))
    return policy