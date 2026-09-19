# training/losses.py
#
# Item 2: an objective that explicitly identifies drift and diffusion,
# instead of only scoring the simulated rollout under MSE.

import math

import torch

from models.neural_sde import NeuralSDE


def euler_transition_nll(
    sde: NeuralSDE,
    traj: torch.Tensor,
    z_f: torch.Tensor,
    z_g: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Teacher-forced Euler-Maruyama transition Gaussian NLL.

    For small dt, the Euler-Maruyama discretisation gives
        dX_t ~= f_phi(X_t, z_f) dt + g_psi(X_t, z_g) sqrt(dt) * eps_t,   eps_t ~ N(0, I)
    so each observed one-step transition delta_t = X_{t+dt} - X_t is scored
    as a sample from a diagonal Gaussian with
        mean = f_phi(X_t, z_f) * dt
        std  = g_psi(X_t, z_g) * sqrt(dt)
    conditioned on the *true* X_t (teacher forcing) rather than a state
    produced by simulating forward from x0 -- that keeps this objective a
    clean per-step identification signal for f_phi/g_psi, uncontaminated by
    the compounding rollout error that the existing trajectory-MSE loss
    already scores separately.

    Parameterisation notes (diagonal covariance, raw std not log-variance):
    NeuralSDE.g already returns a diagonal std-like quantity per state
    dimension via softplus(raw) + 1e-3 (see models/neural_sde.py), so this
    function reuses that output directly as the pre-dt-scaling std rather
    than reinterpreting it as a log-variance or adding a second
    parameterisation. No clamp is applied here beyond what NeuralSDE.g
    already does -- see train_meta.py's smoke test for whether that alone
    is numerically sufficient once dt-scaled.

    Args:
        traj: (B, T, d) ground-truth observed trajectory, T = n_steps + 1
            points spaced by dt (adjacent simulator time points).
        z_f, z_g: (B, zf_dim), (B, zg_dim) latents, one pair per trajectory.
        dt: step size between adjacent points in `traj`. Callers should pass
            the same dt the Euler-Maruyama simulator uses (cfg.time_grid.dt),
            not a separately chosen value.

    Returns:
        Scalar: NLL per transition, summed over state dimensions (diagonal
        covariance) and averaged over batch and time steps.
    """
    B, T, d = traj.shape
    n_transitions = T - 1

    x_t = traj[:, :-1, :].reshape(-1, d)
    x_next = traj[:, 1:, :].reshape(-1, d)
    n = x_t.shape[0]

    zf_exp = z_f.unsqueeze(1).expand(-1, n_transitions, -1).reshape(n, -1)
    zg_exp = z_g.unsqueeze(1).expand(-1, n_transitions, -1).reshape(n, -1)
    t_dummy = torch.zeros(n, device=traj.device)

    drift = sde.f(t_dummy, x_t, zf_exp)                       # (n, d)
    diff_mat = sde.g(t_dummy, x_t, zg_exp)                     # (n, d, d), diagonal
    sigma = torch.diagonal(diff_mat, dim1=-2, dim2=-1)         # (n, d)

    mean = drift * dt
    std = sigma * math.sqrt(dt)
    var = std * std

    delta = x_next - x_t
    nll_per_dim = 0.5 * (torch.log(2.0 * math.pi * var) + (delta - mean) ** 2 / var)
    return nll_per_dim.sum(dim=-1).mean()
