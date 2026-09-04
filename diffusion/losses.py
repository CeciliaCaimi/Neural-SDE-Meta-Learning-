"""Denoising loss w(t) * ||eps - eps_hat||^2.

The three losses of equations (27), (28) and (29) differ only in **which z** is
used and **which query batch** they run on; the loss function itself is the same.
It is therefore implemented once here and reused in all three places.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion.schedule import NoiseSchedule

WEIGHTINGS = ("simple", "snr", "min_snr")


def loss_weight(
    schedule: NoiseSchedule, t: Tensor, kind: str = "simple", gamma: float = 5.0
) -> Tensor:
    """w(t), returned as (B,) for the caller to broadcast.

    simple  : w = 1, the original DDPM eps-prediction objective; most stable at
              low resolution
    snr     : w = SNR(t)
    min_snr : w = min(SNR(t), gamma), the truncation of Hang et al., which stops
              low-noise steps from dominating the gradient
    """
    if kind == "simple":
        return torch.ones_like(t, dtype=torch.float32)
    if kind == "snr":
        return schedule.snr(t, ndim=1).reshape(-1)
    if kind == "min_snr":
        return schedule.snr(t, ndim=1).reshape(-1).clamp(max=gamma)
    raise ValueError(f"unknown weighting '{kind}'; expected one of {WEIGHTINGS}")


def denoising_loss(
    eps_true: Tensor,
    eps_pred: Tensor,
    t: Tensor,
    schedule: NoiseSchedule,
    weighting: str = "simple",
    gamma: float = 5.0,
    reduce: bool = True,
) -> Tensor:
    """w(t) * ||eps - eps_hat||^2, averaged over pixels per sample and then
    weighted by w.

    With reduce=False the per-sample vector (B,) is returned, which the
    diagnostics and stratified statistics use.
    """
    if eps_true.shape != eps_pred.shape:
        raise ValueError(f"shape mismatch: {tuple(eps_true.shape)} vs {tuple(eps_pred.shape)}")
    per_sample = (eps_true - eps_pred).pow(2).flatten(1).mean(1)      # (B,)
    weighted = loss_weight(schedule, t, weighting, gamma) * per_sample
    return weighted.mean() if reduce else weighted
