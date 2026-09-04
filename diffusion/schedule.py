"""Noise schedule.

Corresponding to section 7.2:

    x_t = α_t·x_0 + σ_t·ε ,      ε ~ N(0, I)
    ε*(x_t,t) = −σ_t · s*(x_t,t)                                   (19)

Equation (19) is the bridge of the whole method: adding a basis in eps_hat space is
equivalent to adding one in score space, up to the **known** time scale sigma_t. The model
is implemented in eps_hat (compatible with standard DDPM code) and interpreted as a score.

The schedule is independent of the backbone; swapping U-Net, DiT or any other changes nothing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def linear_betas(n_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)


def cosine_betas(n_steps: int, s: float = 8e-3, max_beta: float = 0.999) -> Tensor:
    """The cosine schedule of Nichol and Dhariwal, usually better than linear at low resolution."""
    t = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64) / n_steps
    ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1.0 - ac[1:] / ac[:-1]
    return betas.clamp(max=max_beta)


BETA_SCHEDULES = {"linear": linear_betas, "cosine": cosine_betas}


class NoiseSchedule(nn.Module):
    """Discrete DDPM schedule. An nn.Module, so its buffers follow .to(device).

    Index convention: t in {0, ..., n_steps-1}, with t=0 the least noisy step.
    """

    def __init__(self, n_steps: int = 1000, kind: str = "cosine", **kwargs) -> None:
        super().__init__()
        if kind not in BETA_SCHEDULES:
            raise ValueError(f"unknown schedule '{kind}'; expected one of {sorted(BETA_SCHEDULES)}")
        self.n_steps = int(n_steps)
        self.kind = kind

        betas = BETA_SCHEDULES[kind](self.n_steps, **kwargs)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alpha", alphas_cumprod.sqrt().float())              # α_t
        self.register_buffer("sigma", (1.0 - alphas_cumprod).sqrt().float())      # σ_t

    # ---- Per-sample coefficient lookup ------------------------------------

    def _gather(self, buf: Tensor, t: Tensor, ndim: int) -> Tensor:
        """Index by t and broadcast to (B, 1, 1, ...) for multiplication with x."""
        out = buf.to(t.device)[t]
        return out.reshape(-1, *([1] * (ndim - 1)))

    def alpha_t(self, t: Tensor, ndim: int = 4) -> Tensor:
        return self._gather(self.alpha, t, ndim)

    def sigma_t(self, t: Tensor, ndim: int = 4) -> Tensor:
        return self._gather(self.sigma, t, ndim)

    def snr(self, t: Tensor, ndim: int = 1) -> Tensor:
        """Signal-to-noise ratio alpha_t^2 / sigma_t^2, used for loss weighting."""
        a, s = self.alpha_t(t, ndim), self.sigma_t(t, ndim)
        return (a ** 2) / (s ** 2).clamp_min(1e-12)

    # ---- eps <-> score conversion, equation (19) --------------------------

    def eps_to_score(self, eps: Tensor, t: Tensor) -> Tensor:
        """s(x_t,t) = −ε̂(x_t,t) / σ_t"""
        return -eps / self.sigma_t(t, eps.dim())

    def score_to_eps(self, score: Tensor, t: Tensor) -> Tensor:
        """ε̂(x_t,t) = −σ_t · s(x_t,t)"""
        return -self.sigma_t(t, score.dim()) * score

    # ---- Timestep sampling ------------------------------------------------

    def sample_t(self, n: int, device=None, generator=None) -> Tensor:
        return torch.randint(0, self.n_steps, (n,), device=device, generator=generator)

    def extra_repr(self) -> str:
        return f"n_steps={self.n_steps}, kind={self.kind}"
