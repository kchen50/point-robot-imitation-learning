import torch
import torch.nn as nn

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

    @staticmethod
    def load(path : str) -> 'Policy':
        policy = Policy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        policy.load_state_dict(torch.load(path))
        return policy

    @staticmethod
    def save(policy : 'Policy', path : str):
        torch.save(policy.state_dict(), path)

    