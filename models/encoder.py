# models/encoder.py

import torch
import torch.nn as nn

class TrajEncoder(nn.Module):
    """
    Trajectory Encoder using a GRU.

    Purpose:
    Reads a sequence of observations (x_t) and compresses them into
    a factorised latent context (z_f, z_g): separate coordinates for
    the drift mechanism and the diffusion mechanism.

    Structure:
    Input (Batch, Time, Dim) -> GRU (Deep) -> Last Hidden State -> two Linear heads -> (z_f, z_g)
    """
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # 1. Recurrent Layer (now deep and regularized)
        self.rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # 2. Separate projection heads on the shared GRU trunk.
        # z_dim is the per-factor dimension: z_f and z_g each have this size.
        self.fc_zf = nn.Linear(hidden_dim, z_dim)
        self.fc_zg = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Tensor of shape (batch, seq_len, x_dim)
        Returns:
            z_f, z_g: each (batch, z_dim) — drift and diffusion latents.
        """
        # Safety Check: Input Dimension
        if x_seq.ndim != 3:
            raise ValueError(f"Encoder input must be (batch, seq, x_dim), got {x_seq.shape}")

        # GRU Forward Pass
        # output: (batch, seq_len, hidden_dim) - features at every step
        # h_n:    (num_layers, batch, hidden_dim) - final hidden state
        _, h_n = self.rnn(x_seq)

        # We take the final hidden state of the LAST stacked layer
        # This represents the deepest abstraction of the sequence.
        h_last = h_n[-1]  # Shape: (batch, hidden_dim)

        # Project to the two factor-specific latent spaces
        z_f = self.fc_zf(h_last)  # Shape: (batch, z_dim)
        z_g = self.fc_zg(h_last)  # Shape: (batch, z_dim)

        return z_f, z_g