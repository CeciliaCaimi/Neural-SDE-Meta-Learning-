
# sde_basis/parameterised_sde.py

import torch
import torch.nn.functional as F

# --- FIX: Use absolute imports instead of relative ('.') ---
# This assumes you are running your code/notebook from the main project root folder.
from sde_basis.basis_functions import drift_basis, diffusion_basis
from sde_basis.parameters import Theta

def drift_true(x: torch.Tensor, theta: Theta) -> torch.Tensor:
    """
    True drift b(x; θ_b) defined via basis expansion.

    NOTE: This implementation assumes *diagonal* drift dependencies.
    The drift of dimension i depends only on the state of dimension i.
    Interactions between dimensions are modeled via the correlated diffusion term.

    Parameters
    ----------
    x : torch.Tensor
        State, shape (..., d)
    theta : Theta
        theta_b : (d, n_drift_basis)

    Returns
    -------
    b : torch.Tensor
        Drift evaluated at x, shape (..., d)
    """
    # Basis features: (..., d, n_b)
    phi_b = drift_basis(x)

    theta_b = theta.theta_b  # (d, n_b)

    # Einstein summation:
    #   Batch dims (...) are preserved.
    #   d matches dimension index.
    #   k sums over basis functions.
    b = torch.einsum("...dk,dk->...d", phi_b, theta_b)
    return b


def sigma_diag_true(
    x: torch.Tensor,
    theta: Theta,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    True *diagonal* diffusion amplitude σ_diag(x; θ_σ).

    Parameters
    ----------
    x : torch.Tensor
        State, shape (..., d)
    theta : Theta
        theta_sigma : (d, n_diffusion_basis)
    eps : float
        Small positive offset to avoid exactly-zero diffusion.

    Returns
    -------
    sigma_diag : torch.Tensor
        Positive diffusion coefficients, shape (..., d).
    """
    # Basis features: (..., d, n_sigma)
    phi_sigma = diffusion_basis(x)

    theta_sigma = theta.theta_sigma  # (d, n_diffusion_basis)

    # Linear combination
    raw = torch.einsum("...dk,dk->...d", phi_sigma, theta_sigma)

    # Ensure strictly positive diffusion via softplus
    sigma_diag = F.softplus(raw) + eps
    return sigma_diag
