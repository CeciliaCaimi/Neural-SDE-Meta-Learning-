"""Forward noising q(x_t | x_0).

    x_t = alpha_t * x_0 + sigma_t * eps,   eps ~ N(0, I)

Independent of both the backbone and the task coordinate -- it depends only on
the NoiseSchedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from diffusion.schedule import NoiseSchedule


@dataclass(frozen=True)
class NoisedBatch:
    x_t: Tensor      # (B, C, H, W) the noised sample
    eps: Tensor      # (B, C, H, W) the true noise: regression target of the loss
    t: Tensor        # (B,) timesteps


def q_sample(
    schedule: NoiseSchedule,
    x0: Tensor,
    t: Tensor | None = None,
    eps: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> NoisedBatch:
    """Noise a batch of clean images. If t / eps are omitted they are sampled;
    supplying them makes the call deterministic, which the tests rely on."""
    if t is None:
        t = schedule.sample_t(x0.shape[0], device=x0.device, generator=generator)
    if eps is None:
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)

    a = schedule.alpha_t(t, x0.dim())
    s = schedule.sigma_t(t, x0.dim())
    return NoisedBatch(x_t=a * x0 + s * eps, eps=eps, t=t)


def to_model_input(x_uint8: Tensor) -> Tensor:
    """uint8 [0, 255] -> float [-1, 1], the standard DDPM input range."""
    return x_uint8.float().div_(127.5).sub_(1.0)


def from_model_output(x: Tensor) -> Tensor:
    """float [-1, 1] -> uint8 [0, 255]."""
    return x.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
