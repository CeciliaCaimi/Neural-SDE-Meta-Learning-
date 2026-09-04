"""Reverse sampling. **Every method shares one sampler**, so sample quality stays comparable.

The sampler depends only on a callable eps_fn(x_t, t) -> eps_hat, so it knows nothing of the
backbone or the coordinate mechanism: changing either leaves this file untouched.

clip_x0 is not cosmetic. Under the cosine schedule alpha_bar[T-1] is about 2.4e-9, so
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t) amplifies the error in
eps_hat by ~2e4 and diverges at high noise. Images clip to [-1, 1]; other domains need a bound.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion.schedule import NoiseSchedule


def _predict_x0(x_t: Tensor, eps: Tensor, ab_t: Tensor, clip_x0: float | None) -> Tensor:
    x0 = (x_t - (1.0 - ab_t).clamp_min(0).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
    return x0 if clip_x0 is None else x0.clamp(-clip_x0, clip_x0)


@torch.no_grad()
def ddpm_sample(
    schedule: NoiseSchedule,
    eps_fn,
    shape: tuple[int, ...],
    device,
    generator: torch.Generator | None = None,
    n_steps: int | None = None,
    clip_x0: float | None = 1.0,
) -> Tensor:
    """DDPM ancestral sampling, using the posterior mean of q(x_{t-1} | x_t, x0_hat) so x0_hat can be clipped."""
    T = schedule.n_steps
    steps = list(range(T - 1, -1, -1)) if n_steps is None else \
        [int(v) for v in torch.linspace(T - 1, 0, n_steps).round().long().tolist()]

    x = torch.randn(shape, device=device, generator=generator)
    ab = schedule.alphas_cumprod.to(device)
    betas = schedule.betas.to(device)

    for i, t in enumerate(steps):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tt)
        ab_t = ab[t]
        t_prev = steps[i + 1] if i + 1 < len(steps) else -1
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.ones((), device=device)

        x0 = _predict_x0(x, eps, ab_t, clip_x0)
        a_t = 1.0 - betas[t]
        coef_x0 = ab_prev.sqrt() * betas[t] / (1.0 - ab_t)
        coef_xt = a_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)
        mean = coef_x0 * x0 + coef_xt * x

        if t_prev >= 0:
            var = (1.0 - ab_prev) / (1.0 - ab_t) * betas[t]
            x = mean + var.clamp_min(0).sqrt() * torch.randn(shape, device=device, generator=generator)
        else:
            x = x0
    return x


@torch.no_grad()
def ddim_sample(
    schedule: NoiseSchedule,
    eps_fn,
    shape: tuple[int, ...],
    device,
    n_steps: int = 50,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    clip_x0: float | None = 1.0,
) -> Tensor:
    """DDIM. Fully deterministic at eta=0 -- use this for reproducible comparisons."""
    T = schedule.n_steps
    ts = [int(v) for v in torch.linspace(T - 1, 0, n_steps).round().long().tolist()]
    ab = schedule.alphas_cumprod.to(device)

    x = torch.randn(shape, device=device, generator=generator)
    for i, t in enumerate(ts):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tt)
        ab_t = ab[t]
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.ones((), device=device)

        x0 = _predict_x0(x, eps, ab_t, clip_x0)
        if t_prev < 0:
            x = x0
            break
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp_min(0).sqrt()
        dir_xt = (1 - ab_prev - sigma ** 2).clamp_min(0).sqrt() * eps
        x = ab_prev.sqrt() * x0 + dir_xt
        if eta > 0:
            x = x + sigma * torch.randn(shape, device=device, generator=generator)
    return x


def make_eps_fn(model, z: Tensor):
    """Wrap a (ScoreModel, z) pair into the eps_fn the sampler expects."""
    def eps_fn(x_t: Tensor, t: Tensor) -> Tensor:
        return model.eps_hat(x_t, t, z.unsqueeze(0).expand(x_t.shape[0], -1))
    return eps_fn
