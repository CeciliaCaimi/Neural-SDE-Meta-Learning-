
#data_gen/simulate_true_sde.py
'''
API Cleanup: Removed dt from the arguments. It is now calculated as T / n_steps to prevent conflicting definitions.

Defensive "Freezing": Added logic to detect invalid particles inside the loop and reset them to 0.0. This prevents NaN or Inf from propagating into the drift calculation of the next step, which saves the GPU from crunching useless numbers and avoids potential errors.

Device Consistency: Explicitly ensures new tensors are created on the correct device.
'''

import torch
from typing import Tuple
from sde_basis.parameters import Theta
from sde_basis.parameterised_sde import drift_true, sigma_diag_true

def simulate_batch(
    theta: Theta,
    L: torch.Tensor,        # (d, d) Cholesky factor
    x0: torch.Tensor,       # (batch_size, d)
    T: float,
    n_steps: int,
    x_max_abs: float,
    generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a batch of trajectories for a SINGLE theta using Euler-Maruyama.

    Features:
    - Pure batch processing for speed.
    - Correlated noise via Cholesky factor L.
    - Stability checks: marks trajectories as invalid if they hit NaNs or bounds.
    - NaN Defense: Resets invalid particles to 0.0 to prevent error propagation.

    Parameters
    ----------
    theta : Theta
        The true SDE parameters.
    L : torch.Tensor
        Cholesky factor of the correlation matrix, shape (d, d).
    x0 : torch.Tensor
        Initial conditions, shape (batch_size, d).
    T : float
        Total time horizon.
    n_steps : int
        Number of discretization steps.
    x_max_abs : float
        Stability threshold. If any dimension |x| > x_max_abs, path is invalid.
    generator : torch.Generator
        Random number generator for reproducibility.

    Returns
    -------
    trajs : torch.Tensor
        Shape (batch_size, n_steps + 1, d).
    valid_mask : torch.Tensor
        Shape (batch_size,). Boolean mask indicating which paths survived.
    """
    batch_size, d = x0.shape
    device = x0.device
    dt = T / n_steps  # Single source of truth for dt
    sqrt_dt = dt ** 0.5
    
    # Pre-allocate trajectory tensor
    # shape: (batch_size, n_steps + 1, d)
    trajs = torch.zeros(batch_size, n_steps + 1, d, device=device)
    trajs[:, 0, :] = x0
    
    x = x0.clone()
    
    # Validity mask (starts all True). 
    # Once False, it stays False (monotonic).
    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Broadcast L for batch matmul: (d, d) -> (1, d, d)
    L_broad = L.unsqueeze(0)

    for k in range(n_steps):
        # 1. Compute Drift & Diffusion
        # Note: If x contains NaNs or Infs from previous step, these will output NaNs.
        # We handle this by freezing invalid x below.
        b = drift_true(x, theta)
        sigma = sigma_diag_true(x, theta)
        
        # 2. Sample Noise
        # xi: (batch, d, 1) to allow matmul with L
        xi = torch.randn(batch_size, d, 1, generator=generator, device=device)
        
        # 3. Correlate noise: L @ xi
        # (1, d, d) @ (batch, d, 1) -> (batch, d, 1)
        noise_corr = torch.matmul(L_broad, xi).squeeze(-1) # (batch, d)
        
        # 4. Euler-Maruyama Step
        dx = b * dt + sigma * noise_corr * sqrt_dt
        x_next = x + dx
        
        # 5. Stability Check
        # Check NaNs or Infinities
        is_finite = torch.isfinite(x_next).all(dim=1)
        # Check bounds
        is_bounded = (x_next.abs().max(dim=1).values <= x_max_abs)
        
        # Update valid_mask (Logic: Must have BEEN valid AND be valid NOW)
        still_valid = is_finite & is_bounded
        valid_mask = valid_mask & still_valid

        # 6. Defensive Freeze
        # If a particle is invalid, force it to 0.0.
        # This prevents 'Inf' from exploding in the x^3 term of drift_true 
        # in the next iteration, which can cause CUDA errors or NaN propagation.
        # We only strictly need valid trajectories, so the values here don't matter.
        if (~valid_mask).any():
            x_next[~valid_mask] = 0.0
        
        # 7. Update and Store
        x = x_next
        trajs[:, k + 1, :] = x

    return trajs, valid_mask