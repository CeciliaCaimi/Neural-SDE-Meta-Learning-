"""Deterministic CIFAR-100-C style corruptions (sections 3.3, 12.3 and A.6).

source = the clean image, target = the corrupted image; the corruption *is* the S->T shift.
A.6 requires determinism and a stored transformation id: given the global image id, the
corruption type and the severity, the result must be exactly reproducible. Corruptions that
involve randomness (noise) therefore derive their seed from (gid, type, severity).

Pure torch throughout, acting on float tensors in [-1, 1] and batched on the GPU.
The three the document suggests to begin with: blur, noise and contrast.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

CORRUPTIONS = ("gaussian_blur", "gaussian_noise", "contrast")
CORRUPTION_ID = {name: i for i, name in enumerate(CORRUPTIONS)}


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> Tensor:
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur. x: (N, C, H, W) in [-1, 1]. Deterministic."""
    radius = max(1, int(math.ceil(3 * sigma)))
    k1 = _gaussian_kernel1d(sigma, radius, x.device, x.dtype)
    C = x.shape[1]
    kh = k1.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    kw = k1.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    x = torch.nn.functional.conv2d(x, kh, padding=(radius, 0), groups=C)
    x = torch.nn.functional.conv2d(x, kw, padding=(0, radius), groups=C)
    return x


def gaussian_noise(x: Tensor, std: float, gids: Tensor) -> Tensor:
    """Per-image Gaussian noise seeded from the global image id, so an image always gets the same noise."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        g = torch.Generator(device=x.device).manual_seed(int(gids[i]) * 97 + 12345)
        out[i] = x[i] + std * torch.randn(x[i].shape, generator=g, device=x.device, dtype=x.dtype)
    return out.clamp(-1.0, 1.0)


def contrast(x: Tensor, factor: float) -> Tensor:
    """Reduce contrast by a factor below 1. Deterministic; each image shrinks toward its own mean."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * factor + mean).clamp(-1.0, 1.0)


# Default strength of each corruption at each severity (1..5, with 3 the middle)
_DEFAULT_PARAM = {
    "gaussian_blur": {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0},          # sigma
    "gaussian_noise": {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.26, 5: 0.38},    # std on the [-1,1] scale
    "contrast": {1: 0.75, 2: 0.6, 3: 0.45, 4: 0.3, 5: 0.2},             # factor
}


def apply_corruption(x: Tensor, name: str, severity: int = 3,
                     gids: Tensor | None = None) -> Tensor:
    """Single entry point. x: (N, C, H, W) in [-1, 1]; returns a tensor of the same shape."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption '{name}'; expected one of {CORRUPTIONS}")
    p = _DEFAULT_PARAM[name][severity]
    if name == "gaussian_blur":
        return gaussian_blur(x, p)
    if name == "gaussian_noise":
        if gids is None:
            raise ValueError("gaussian_noise needs gids in order to be deterministic")
        return gaussian_noise(x, p, gids)
    return contrast(x, p)
