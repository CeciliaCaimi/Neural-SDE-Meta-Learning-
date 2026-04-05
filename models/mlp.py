# models/mlp.py

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron.

    Used for:
    - Drift Network (f)
    - Diffusion Network (g)
    - Forecasting Head

    Structure:
      Input -> [Linear -> (LayerNorm) -> Activation] x N -> Linear -> Output
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int, 
        activation: Optional[nn.Module] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        
        if activation is None:
            activation = nn.SiLU()  # default, but created per MLP instance
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(activation)
            in_dim = h_dim
            
        # Output layer (no norm / activation)
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
