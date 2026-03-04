import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim * 2)

    def _params(self, state):
        """
        Returns:
          mean:     (..., A)
          cov_diag: (..., A)   (diagonal of covariance, NOT std)
        """
        x = F.relu(self.fc1(state))
        value = self.fc2(x)  # (..., 2A)

        A = self.action_dim
        mean_raw = value[..., :A]      # (..., A)
        cov_raw  = value[..., A:]      # (..., A)

        # Keep your existing mean squash + scale
        mean = torch.tanh(mean_raw) * 0.01

        # Keep your existing covariance parameterization but add eps for stability
        cov_diag = 0.01 * torch.sigmoid(cov_raw) + 1e-8  # (..., A)

        return mean, cov_diag

    def _dist(self, state):
        mean, cov_diag = self._params(state)
        std = torch.sqrt(cov_diag)  # convert covariance diag -> std diag
        base = torch.distributions.Normal(mean, std)
        return torch.distributions.Independent(base, 1), cov_diag

    @torch.no_grad()
    def sample_action(self, state, noise=False):
        dist, cov_diag = self._dist(state)
        a = dist.sample()

        # Preserve your "return cov[0][0]" behavior as a scalar:
        # - if batched, take the first item's first action-dim covariance
        # - if unbatched, take the first action-dim covariance
        if cov_diag.dim() >= 2:
            cov00 = cov_diag[0, 0]
        else:
            cov00 = cov_diag[0]

        return a, cov00

    def forward(self, state, action):
        dist, _ = self._dist(state)
        return dist.log_prob(action)

class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Value function (single output)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value