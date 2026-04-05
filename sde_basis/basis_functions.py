# sde_basis/basis_functions.py

import torch

"""
Basis functions φ_b and φ_σ used to define the *true* SDEs.

We treat each dimension independently: for x ∈ R^d, we build per-coordinate
features and later linearly combine them with θ coefficients.
"""

def drift_basis(x: torch.Tensor) -> torch.Tensor:
    """
    φ_b(x): ... x d -> ... x d x 8

    Basis: [1, x, x^2, x^3, sin(x), cos(x), tanh(x), x^4]
    """
    phi0 = torch.ones_like(x)
    phi1 = x
    phi2 = x ** 2
    phi3 = x ** 3
    phi4 = torch.sin(x)
    phi5 = torch.cos(x)
    phi6 = torch.tanh(x)
    phi7 = x ** 4

    phi = torch.stack([phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7], dim=-1)
    return phi


def diffusion_basis(x: torch.Tensor) -> torch.Tensor:
    """
    φ_σ(x): ... x d -> ... x d x 3

    Basis: [1, |x|, sqrt(|x| + eps)]

    Parameters
    ----------
    x : torch.Tensor
        Current state.

    Returns
    -------
    phi : torch.Tensor
        Shape (*x.shape, 3).
    """
    # Hardcoded safety epsilon for the basis calculation itself
    # (distinct from the diffusion_eps in config which shifts the result)
    eps = 1e-6
    absx = torch.abs(x)

    phi0 = torch.ones_like(x)
    phi1 = absx
    phi2 = torch.sqrt(absx + eps)

    phi = torch.stack([phi0, phi1, phi2], dim=-1)
    return phi