"""Shared set encoder r_psi (section 8.2).

    r_ψ(D) = ρ_ψ( (1/m) Σ_i h_ψ(x_i) )                                       (22)

**Source and target share one encoder**; there is no separate branch. The document is
explicit: "There is no separate source-specific or target-specific representation network."

Mean pooling is the default (permutation invariant, simple, exactly streamable); the
document calls it a validation choice, not a contribution. If diagnostics show z failing
to separate the domains, change the pooling (second moments, attention) before the encoder.

Both implementations share one interface (per_element / forward / forward_sets / stream),
so the meta layer never needs to know whether the domain is images or vectors:
  - ConvSetEncoder   : images (stages 2 and 3)
  - VectorSetEncoder : R^d vectors (the GMM points of stage 1)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_STD_EPS = 1e-5      # floor inside the sqrt of mean_std pooling, keeping gradients finite


class SetEncoderBase(nn.Module):
    """Interface: subclasses implement per_element; the base class supplies the rest."""

    k: int

    def per_element(self, x: Tensor) -> Tensor:
        """h_psi: (N, ...) -> (N, feature_dim)."""
        raise NotImplementedError

    pooling: str = "mean"          # "mean" | "mean_std", set by the subclass

    def pool_head(self, pooled: Tensor) -> Tensor:
        """rho_psi: (pool_dim,) -> (k,). pool_dim = feature_dim, doubled for mean_std."""
        raise NotImplementedError

    def _pool(self, feats: Tensor) -> Tensor:
        """Pool per-element features (N, d) into a single vector.

        mean     : first moment, (d,)
        mean_std : first and second moments concatenated, (2d,), the std in population
                   form, hence 0 when N=1. Permutation invariant and streamable.
        """
        mu = feats.mean(dim=0)
        if self.pooling == "mean":
            return mu
        var = (feats.pow(2).mean(dim=0) - mu.pow(2)).clamp_min(0.0)
        # Add eps before the sqrt: a single-element set (K_T = 1) has zero variance, and
        # the gradient of sqrt(0) is inf, which turns the model to NaN within a few steps.
        return torch.cat([mu, (var + _STD_EPS).sqrt()], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """One set -> one coordinate. (N, ...) -> (k,)."""
        return self.pool_head(self._pool(self.per_element(x)))

    def forward_sets(self, sets: list[Tensor]) -> Tensor:
        """A batch of sets -> (B, k). Sets may differ in size (source large, target small)."""
        return torch.stack([self.forward(s) for s in sets], dim=0)

    def stream(self) -> "StreamingPool":
        return StreamingPool(self)

    # Backwards-compatible alias
    def per_image(self, x: Tensor) -> Tensor:
        return self.per_element(x)


class ConvSetEncoder(SetEncoderBase):
    """Image domain: 32 -> 16 -> 8 -> 4 strided convolutions, then a global average."""

    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 32,
        width: int = 64,
        feature_dim: int = 256,
        k: int = 16,
        hidden: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feature_dim = int(feature_dim)
        self.pooling = pooling
        pool_dim = feature_dim * (2 if pooling == "mean_std" else 1)

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.SiLU(),
            )

        self.h = nn.Sequential(
            block(image_channels, width),
            block(width, width * 2),
            block(width * 2, width * 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, feature_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.k)
        )

    def per_element(self, x: Tensor) -> Tensor:
        return self.h(x)

    def pool_head(self, pooled: Tensor) -> Tensor:
        return self.rho(pooled)


class VectorSetEncoder(SetEncoderBase):
    """Vector domain (DeepSets). Accepts (N, d) or (N, d, 1, 1)."""

    def __init__(
        self,
        dim: int = 2,
        feature_dim: int = 128,
        k: int = 16,
        hidden: int = 128,
        depth: int = 3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.dim = int(dim)
        self.feature_dim = int(feature_dim)
        self.pooling = pooling
        pool_dim = feature_dim * (2 if pooling == "mean_std" else 1)

        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, feature_dim)]
        self.h = nn.Sequential(*layers)
        self.rho = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.k)
        )

    def per_element(self, x: Tensor) -> Tensor:
        return self.h(x.reshape(x.shape[0], -1))

    def pool_head(self, pooled: Tensor) -> Tensor:
        return self.rho(pooled)


# Backwards compatibility: SetEncoder in older code means the image version
SetEncoder = ConvSetEncoder


class StreamingPool:
    """Accumulate sum h_psi(x) and a count in chunks, passing rho_psi once at the end.

    Because mean pooling is linear, chunked accumulation is numerically identical to
    feeding everything at once. Used only for the **deployment** source anchor (no
    gradients); training must retain gradients, so M_S there is bounded by memory.
    """

    def __init__(self, encoder: SetEncoderBase) -> None:
        self.encoder = encoder
        self._sum: Tensor | None = None
        self._sqsum: Tensor | None = None      # for mean_std: sum of h^2
        self._n = 0

    @torch.no_grad()
    def add(self, x: Tensor) -> "StreamingPool":
        feats = self.encoder.per_element(x)
        s = feats.sum(dim=0)
        self._sum = s if self._sum is None else self._sum + s
        if self.encoder.pooling == "mean_std":
            sq = feats.pow(2).sum(dim=0)
            self._sqsum = sq if self._sqsum is None else self._sqsum + sq
        self._n += int(x.shape[0])
        return self

    @torch.no_grad()
    def finalize(self) -> Tensor:
        if self._n == 0:
            raise RuntimeError("streaming pool is empty; call add() first")
        mu = self._sum / self._n
        if self.encoder.pooling == "mean":
            pooled = mu
        else:
            var = (self._sqsum / self._n - mu.pow(2)).clamp_min(0.0)
            pooled = torch.cat([mu, (var + _STD_EPS).sqrt()], dim=-1)
        return self.encoder.pool_head(pooled)

    @property
    def count(self) -> int:
        return self._n
