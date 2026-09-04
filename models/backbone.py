"""Swappable diffusion backbone interface.

This is the single seam between the **meta layer** and the **diffusion network**.

The design follows the architectural split of section 10 of the source document:

    1. Shared backbone   : a U-Net takes (x_t, t) and emits shared features H
    2. base + basis head : base emits eps_hat_0; basis emits k x C_img channels
                                 —— "This uses one U-Net, not k separate denoisers"
    3. Shared set encoder
    4. relation transport

Item 1 lives here; items 2-4 live in the meta layer (score_model / set_encoder /
transport). **Neither head nor the z mechanism belongs to the backbone**, or else

────────────────────────────────────────────────────────────────────────
Everything required to swap in a different backbone
────────────────────────────────────────────────────────────────────────

    @register_backbone("my_dit")
    class MyDiT(DiffusionBackbone):
        @property
        def spec(self) -> BackboneSpec: ...
        def forward_features(self, x_t, t) -> Tensor: ...   # (B, C_feat, H, W)

Then run `check_backbone(MyDiT(...))` once. ScoreModel / SetEncoder / Transport /
the training loop / the diagnostics need no change at all.

The contract:
  - forward_features must emit a **spatial feature map at input resolution**,
    (B, C_feat, H, W). Token-based backbones (DiT and the like) unpatchify internally.
  - It must genuinely depend on t: the same x at different t gives different features.
  - Same input, same output (deterministic in eval mode).
  - If the backbone has its own eps_hat head (a pretrained DDPM, say), override
    forward_eps and set spec.provides_eps=True; ScoreModel then reuses it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BackboneSpec:
    """Everything a backbone declares to the meta layer; the only numbers it reads."""

    feature_channels: int     # C_feat: input channels of both heads
    image_channels: int       # C_img: image channels (CIFAR = 3)
    image_size: int           # H = W
    provides_eps: bool = False  # whether the backbone has its own eps_hat head

    def __post_init__(self) -> None:
        for name in ("feature_channels", "image_channels", "image_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"BackboneSpec.{name} must be positive")


class DiffusionBackbone(nn.Module, ABC):
    """Any diffusion network plugged into this framework implements this class."""

    @property
    @abstractmethod
    def spec(self) -> BackboneSpec:
        """Declare feature channels, image shape, and whether an eps_hat head exists."""

    @abstractmethod
    def forward_features(self, x_t: Tensor, t: Tensor) -> Tensor:
        """(B, C_img, H, W) x (B,) -> (B, C_feat, H, W)."""

    def forward_eps(self, x_t: Tensor, t: Tensor, feats: Tensor | None = None) -> Tensor | None:
        """The backbone's own eps_hat prediction. None means it has none, in which

        case ScoreModel builds its own base head. Override this (and set
        spec.provides_eps=True) to reuse a pretrained eps_hat, training only the basis head.
        """
        return None

    # nn.Module.forward follows the feature path, so the object works as a plain module
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:  # noqa: D102
        return self.forward_features(x_t, t)


# ---------------------------------------------------------------------------
# Registry: lets the config name a backbone by string, without importing the class
# ---------------------------------------------------------------------------

BACKBONES: dict[str, type[DiffusionBackbone]] = {}


def register_backbone(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in BACKBONES:
            raise KeyError(f"backbone name '{name}' already taken by {BACKBONES[name].__name__}")
        if not issubclass(cls, DiffusionBackbone):
            raise TypeError(f"{cls.__name__} must subclass DiffusionBackbone")
        BACKBONES[name] = cls
        return cls
    return deco


def build_backbone(name: str, **kwargs) -> DiffusionBackbone:
    if name not in BACKBONES:
        raise KeyError(f"unregistered backbone '{name}'; registered: {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)


# ---------------------------------------------------------------------------
# Contract check: run this first after writing a new backbone
# ---------------------------------------------------------------------------

class BackboneContractError(AssertionError):
    """The backbone does not satisfy the interface contract."""


def check_backbone(backbone: DiffusionBackbone, batch: int = 2, verbose: bool = True) -> None:
    """Verify a backbone implementation against the contract, raising on failure."""
    spec = backbone.spec
    if not isinstance(spec, BackboneSpec):
        raise BackboneContractError("spec must return a BackboneSpec")

    was_training = backbone.training
    backbone.eval()
    try:
        x = torch.randn(batch, spec.image_channels, spec.image_size, spec.image_size)
        t = torch.tensor([0, spec_max_t(backbone)] if batch == 2
                         else [0] * batch, dtype=torch.long)[:batch]

        with torch.no_grad():
            feats = backbone.forward_features(x, t)

        want = (batch, spec.feature_channels, spec.image_size, spec.image_size)
        if tuple(feats.shape) != want:
            raise BackboneContractError(
                f"forward_features shape should be {want}, got {tuple(feats.shape)}. "
                "Token-based backbones must unpatchify to a spatial map internally."
            )
        if not torch.isfinite(feats).all():
            raise BackboneContractError("forward_features output contains NaN/Inf")

        # Determinism
        with torch.no_grad():
            again = backbone.forward_features(x, t)
        if not torch.allclose(feats, again, atol=0, rtol=0):
            raise BackboneContractError("same input gave different outputs in eval mode")

        # Sensitivity to t.
        # Note: many implementations (guided-diffusion among them) zero-initialise the
        # output convolution of every residual branch, making the network exactly the
        # identity **at initialisation** -- t cannot propagate, though the structure is
        # sound. So perturb the parameters and retry; only then is insensitivity a failure.
        if not _t_sensitive(backbone, x, batch):
            with _perturbed(backbone, std=1e-2):
                if not _t_sensitive(backbone, x, batch):
                    raise BackboneContractError(
                        "features do not depend on t even after perturbation -- time embedding not wired up?"
                    )

        # Consistency of a built-in eps_hat head
        eps = backbone.forward_eps(x, t)
        if spec.provides_eps:
            if eps is None:
                raise BackboneContractError("spec.provides_eps=True but forward_eps returned None")
            want_eps = (batch, spec.image_channels, spec.image_size, spec.image_size)
            if tuple(eps.shape) != want_eps:
                raise BackboneContractError(f"forward_eps shape should be {want_eps}, got {tuple(eps.shape)}")
        elif eps is not None:
            raise BackboneContractError("forward_eps returned a value but spec.provides_eps=False")

        if verbose:
            n = sum(p.numel() for p in backbone.parameters())
            print(
                f"  [OK] {type(backbone).__name__}: features {want}, "
                f"provides_eps={spec.provides_eps}, params {n/1e6:.2f}M"
            )
    finally:
        backbone.train(was_training)


def spec_max_t(backbone: DiffusionBackbone) -> int:
    """A large timestep used by the contract check to probe sensitivity to t."""
    return int(getattr(backbone, "max_timestep", 999))


def _t_sensitive(backbone: DiffusionBackbone, x: Tensor, batch: int) -> bool:
    t_a = torch.zeros(batch, dtype=torch.long)
    t_b = torch.full((batch,), spec_max_t(backbone), dtype=torch.long)
    with torch.no_grad():
        fa = backbone.forward_features(x, t_a)
        fb = backbone.forward_features(x, t_b)
    return not torch.allclose(fa, fb, atol=1e-6)


class _perturbed:
    """Temporarily perturb all float parameters, restoring them exactly on exit."""

    def __init__(self, module: nn.Module, std: float) -> None:
        self.module, self.std = module, std
        self.saved: list[tuple[nn.Parameter, Tensor]] = []

    def __enter__(self) -> None:
        with torch.no_grad():
            for p in self.module.parameters():
                if p.is_floating_point():
                    self.saved.append((p, p.detach().clone()))
                    p.add_(torch.randn_like(p) * self.std)

    def __exit__(self, *exc) -> None:
        with torch.no_grad():
            for p, original in self.saved:
                p.copy_(original)
        self.saved.clear()
