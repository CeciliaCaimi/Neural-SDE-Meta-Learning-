# models/head.py

import torch
import torch.nn as nn
from models.mlp import MLP

class ForecastHead(nn.Module):
    """
    Forecasting Head.
    
    Takes the final state x_T and latent context z,
    outputs a prediction in R^{x_dim}.
    """
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        
        input_dim = x_dim + z_dim
        
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=x_dim,
            use_layernorm=False,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Shape checks
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, x_dim), got {x.shape}")
        if z.ndim not in (1, 2):
            raise ValueError(f"z must be 1D or 2D, got {z.shape}")

        # Broadcast / align z
        if z.dim() == 1:
            z_in = z.unsqueeze(0).expand(x.size(0), -1)
        elif z.size(0) == 1 and x.size(0) > 1:
            z_in = z.expand(x.size(0), -1)
        else:
            if z.size(0) != x.size(0):
                raise ValueError(
                    f"Batch mismatch: x batch={x.size(0)}, z batch={z.size(0)}"
                )
            z_in = z

        inp = torch.cat([x, z_in], dim=-1)
        return self.net(inp)
