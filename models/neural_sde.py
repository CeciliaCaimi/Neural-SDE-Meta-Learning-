# models/neural_sde.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP


class NeuralSDE(nn.Module):
    """
    Stationary Neural SDE with Context Injection.

    dY_t = f(t, Y_t, z) dt + g(t, Y_t, z) dW_t
    """

    # REQUIRED by torchsde
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        input_dim = x_dim + z_dim

        # Drift network f
        self.drift_net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=x_dim,
            use_layernorm=True,
        )

        # Diffusion network g (diagonal)
        self.diff_net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=x_dim,
            use_layernorm=True,
        )

    @staticmethod
    def _prepare_inputs(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Make y, z batch-compatible and concatenate.
        Handles:
          - y: (x_dim,) or (batch, x_dim)
          - z: (z_dim,), (1, z_dim) or (batch, z_dim)
        """
        # Ensure y has batch dim
        if y.ndim == 1:
            y = y.unsqueeze(0)  # (1, x_dim)

        # Broadcast / expand z
        if z.ndim == 1:
            z = z.unsqueeze(0).expand(y.size(0), -1)          # (batch, z_dim)
        elif z.size(0) == 1 and y.size(0) > 1:
            z = z.expand(y.size(0), -1)                       # (batch, z_dim)

        return torch.cat([y, z], dim=-1)  # (batch, x_dim + z_dim)

    def f(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Drift: f(t, y, z) -> (batch, x_dim)
        t is ignored (stationary SDE) but required by torchsde.
        """
        inp = self._prepare_inputs(y, z)
        return self.drift_net(inp)

    def g(self, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Diffusion: g(t, y, z) -> (batch, x_dim, x_dim) diagonal matrix.
        """
        inp = self._prepare_inputs(y, z)
        raw_sigma = self.diff_net(inp)             # (batch, x_dim)
        sigma = F.softplus(raw_sigma) + 1e-3       # (batch, x_dim)
        return torch.diag_embed(sigma)             # (batch, x_dim, x_dim)
