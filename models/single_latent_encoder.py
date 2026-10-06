# models/single_latent_encoder.py
#
# Item 6, phase 3, arm 1 (single_z): the pre-item-1 architecture, where ONE
# latent is shared by drift, diffusion and the forecast head, instead of
# the factorised (z_f, z_g) pair models/encoder.py's TrajEncoder produces.
#
# This is a NEW class, not a modification of TrajEncoder -- the factorised
# default (TrajEncoder, two independent projection heads) is untouched.
# Same GRU trunk/hyperparameter conventions as TrajEncoder, just one
# projection head instead of two, so a single forward pass produces a
# single z that gets passed as BOTH the "z_f" and "z_g" argument at every
# call site that uses it (see adaptation/compositional_arms.py's encode()),
# which is what actually ties drift-adaptation and diffusion-adaptation
# together (||Delta z_f|| == ||Delta z_g|| by construction, since there is
# only one latent and therefore only one delta) -- not merely initializing
# two separate latents to the same value, which would NOT stay tied once
# Adam/gradient updates started touching them independently.

import torch
import torch.nn as nn


class SingleLatentEncoder(nn.Module):
    """Trajectory encoder producing ONE latent z (not a z_f/z_g pair).

    Structure identical to models.encoder.TrajEncoder's GRU trunk, with a
    single projection head.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_z = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, seq_len, x_dim)
        Returns:
            z: (batch, z_dim) -- the single shared latent.
        """
        if x_seq.ndim != 3:
            raise ValueError(f"Encoder input must be (batch, seq, x_dim), got {x_seq.shape}")
        _, h_n = self.rnn(x_seq)
        h_last = h_n[-1]
        return self.fc_z(h_last)
