"""Backbone for vector-valued data (the R^2 points of stage 1).

A d-dimensional vector is treated as a (B, d, 1, 1) "image", so that the
DiffusionBackbone contract, ScoreModel, both heads, the transport network and
the training loop need **not a single line of change**: stage 1 and stage 3 run
down the same code path. That is itself a test of the abstraction.

As with the U-Net, this emits features only, never eps_hat.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.backbone import BackboneSpec, DiffusionBackbone, register_backbone
from models.unet import timestep_embedding


class _ResMLP(nn.Module):
    """Time-conditioned residual MLP block; the timestep enters as a
    scale-shift, mirroring the U-Net ResBlock."""

    def __init__(self, dim: int, temb_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.emb = nn.Linear(temb_dim, 2 * dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.emb(temb).chunk(2, dim=-1)
        return h + self.net(self.norm(h) * (1 + scale) + shift)


@register_backbone("mlp_vector")
class MLPVectorBackbone(DiffusionBackbone):
    def __init__(
        self,
        image_channels: int = 2,
        hidden: int = 256,
        depth: int = 4,
        feature_channels: int = 128,
        max_timestep: int = 999,
    ) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        self.hidden = hidden
        self._spec = BackboneSpec(
            feature_channels=feature_channels,
            image_channels=image_channels,
            image_size=1,                 # vector data: the "spatial" extent is 1x1
            provides_eps=False,
        )
        temb_dim = hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.inp = nn.Linear(image_channels, hidden)
        self.blocks = nn.ModuleList(_ResMLP(hidden, temb_dim) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, feature_channels))

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def forward_features(self, x_t: Tensor, t: Tensor) -> Tensor:
        b = x_t.shape[0]
        h = self.inp(x_t.reshape(b, -1))
        temb = self.time_mlp(timestep_embedding(t, self.hidden))
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(h).reshape(b, -1, 1, 1)     # (B, C_feat, 1, 1)
