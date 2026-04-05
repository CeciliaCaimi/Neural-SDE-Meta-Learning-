# quick_sanity.py

import torch

from config.base_config import cfg
from sde_basis.basis_functions import drift_basis, diffusion_basis
from sde_basis.parameters import Theta
from sde_basis.covariance import sample_correlation
from sde_basis.parameterised_sde import drift_true, sigma_diag_true


def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    d = cfg.basis.x_dim
    n_b = cfg.basis.n_drift_basis
    n_s = cfg.basis.n_diffusion_basis

    td = cfg.theta_dist

    # -----------------------------
    # 1. Sample a random Theta
    # -----------------------------
    # Drift coefficients θ_b ~ N(mean, std * scale_per_basis)
    theta_b = torch.randn(d, n_b, device=device) * td.drift_std_train + td.drift_mean_train
    drift_scales = torch.tensor(td.drift_scales, device=device).view(1, -1)  # (1, n_b)
    theta_b = theta_b * drift_scales  # broadcast to (d, n_b)

    # Diffusion coefficients θ_σ ~ N(mean, std * scale_per_basis)
    theta_sigma = torch.randn(d, n_s, device=device) * td.diffusion_std_train + td.diffusion_mean_train
    diff_scales = torch.tensor(td.diffusion_scales, device=device).view(1, -1)  # (1, n_s)
    theta_sigma = theta_sigma * diff_scales

    theta = Theta(theta_b=theta_b, theta_sigma=theta_sigma, id="sanity")

    print("Theta shapes:")
    print("  theta_b:", theta.theta_b.shape)
    print("  theta_sigma:", theta.theta_sigma.shape)

    # -----------------------------
    # 2. Basis functions sanity
    # -----------------------------
    x = torch.randn(4, d, device=device)  # batch=4, dim=d

    phi_b = drift_basis(x)
    phi_s = diffusion_basis(x)

    print("\nBasis shapes:")
    print("  drift_basis:", phi_b.shape)      # (4, d, n_b)
    print("  diffusion_basis:", phi_s.shape)  # (4, d, n_s)

    # -----------------------------
    # 3. Drift / diffusion sanity
    # -----------------------------
    b = drift_true(x, theta)
    sigma_diag = sigma_diag_true(x, theta)

    print("\nDrift / diffusion:")
    print("  b shape:", b.shape)              # (4, d)
    print("  sigma_diag shape:", sigma_diag.shape)
    print("  drift has NaN?  ", torch.isnan(b).any().item())
    print("  sigma min/max   ", sigma_diag.min().item(), sigma_diag.max().item())

    # -----------------------------
    # 4. Correlation matrix sanity
    # -----------------------------
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    R, L = sample_correlation(d=d, rng=gen, device=device)

    print("\nCorrelation matrix:")
    print("  R shape:", R.shape)
    print("  L shape:", L.shape)
    eigs = torch.linalg.eigvalsh(R)
    print("  eigenvalues min/max:", eigs.min().item(), eigs.max().item())

    # -----------------------------
    # 5. Short Euler–Maruyama rollout (diag noise only)
    #    (just to check stability & no NaNs)
    # -----------------------------
    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    dt = cfg.time_grid.dt

    x_t = torch.randn(d, device=device) * 0.5  # one trajectory
    x_t0 = x_t.clone()

    blew_up = False
    for n in range(n_steps):
        b_t = drift_true(x_t, theta)           # (d,)
        sigma_diag_t = sigma_diag_true(x_t, theta)  # (d,)
        dW = torch.randn(d, device=device) * (dt ** 0.5)

        # For this sanity test we ignore correlation and just use diagonal noise
        x_t = x_t + b_t * dt + sigma_diag_t * dW

        if x_t.abs().max() > cfg.stability.max_state_abs:
            print(f"\n[EM] Blow-up at step {n} |x|={x_t.abs().max().item():.3f}")
            blew_up = True
            break

    if not blew_up:
        print("\n[EM] Simulation completed without blow-up.")
        print("  initial |x| max:", x_t0.abs().max().item())
        print("  final   |x| max:", x_t.abs().max().item())


if __name__ == "__main__":
    main()
