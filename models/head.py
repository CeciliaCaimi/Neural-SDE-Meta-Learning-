# models/head.py

import torch
import torch.nn as nn
from models.mlp import MLP

class ForecastHead(nn.Module):
    """
    Forecasting Head.

    Takes the final state x_T and the factorised latent context
    (z_f, z_g), outputs a prediction in R^{x_dim}.
    """
    def __init__(self, x_dim: int, zf_dim: int, zg_dim: int, hidden_dim: int):
        super().__init__()

        input_dim = x_dim + zf_dim + zg_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=x_dim,
            use_layernorm=False,
        )

    @staticmethod
    def _align(z: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        if z.ndim not in (1, 2):
            raise ValueError(f"{name} must be 1D or 2D, got {z.shape}")

        if z.dim() == 1:
            return z.unsqueeze(0).expand(batch, -1)
        if z.size(0) == 1 and batch > 1:
            return z.expand(batch, -1)
        if z.size(0) != batch:
            raise ValueError(
                f"Batch mismatch: expected batch={batch}, {name} batch={z.size(0)}"
            )
        return z

    def forward(self, x: torch.Tensor, z_f: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        # Shape checks
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, x_dim), got {x.shape}")

        # Broadcast / align each latent independently against x's batch dim
        z_f_in = self._align(z_f, x.size(0), "z_f")
        z_g_in = self._align(z_g, x.size(0), "z_g")

        inp = torch.cat([x, z_f_in, z_g_in], dim=-1)
        return self.net(inp)
