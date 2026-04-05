# sde_basis/covariance.py

import torch

def sample_correlation(
    d: int,
    rng: torch.Generator,
    device: str,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random SPD *Correlation* matrix R and its Cholesky factor L.

    We normalize diagonals to 1 so that this matrix purely encodes
    correlations (rho). The actual volatility amplitudes are determined
    separately by the diffusion basis functions.

    Steps:
      1. Draw random Gaussian matrix A.
      2. Form Raw Covariance Sigma = A A^T.
      3. Normalize diagonals to 1 -> Correlation matrix R.
      4. Add jitter for numerical stability.
      5. Compute Cholesky L s.t. R = L L^T.

    Returns
    -------
    corr : (d, d)        SPD correlation matrix (diagonals approx 1.0)
    chol : (d, d)        Lower-triangular Cholesky factor
    """
    # 1. Random Gaussian matrix
    A = torch.randn(d, d, generator=rng, device=device)

    # 2. SPD raw covariance
    Sigma = A @ A.T

    # 3. Normalize to Correlation Matrix (diagonals = 1)
    diag = torch.diag(Sigma)
    # Safe inverse sqrt handling
    inv_sqrt_diag = torch.where(
        diag > 0,
        diag.rsqrt(),
        torch.ones_like(diag)
    )
    D = torch.diag(inv_sqrt_diag)
    corr = D @ Sigma @ D

    # 4. Symmetrise numerically
    corr = 0.5 * (corr + corr.T)

    # 5. Add jitter (crucial for Cholesky stability on GPU)
    corr = corr + jitter * torch.eye(d, device=device)

    # 6. Cholesky factor
    chol = torch.linalg.cholesky(corr)

    return corr, chol