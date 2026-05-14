"""Latent Interpolation / Manifold Traversal Figure.

Interpolates between latent codes of two test tasks and visualises
trajectory transitions at each interpolation step.

Outputs:
    results/latent_interpolation.csv
    plots/latent_interpolation.png
"""
import os, sys, torch, numpy as np, matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import simulate_neural_sde_batch
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories, apply_scaler_to_trajectories

N_INTERP = 7  # number of interpolation steps (including endpoints)
TASK_PAIRS = [("testA_000", "testC_000"), ("testB_005", "testC_010")]


def slerp(z1, z2, t):
    """Spherical linear interpolation."""
    z1_n = z1 / (z1.norm() + 1e-8)
    z2_n = z2 / (z2.norm() + 1e-8)
    omega = torch.acos(torch.clamp((z1_n * z2_n).sum(), -1, 1))
    if omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) / torch.sin(omega)) * z1 + \
           (torch.sin(t * omega) / torch.sin(omega)) * z2


def load_task(task_id, data_dir="data/test_trajectories"):
    path = f"{data_dir}/{task_id}/support"
    files = sorted(os.listdir(path))[:5]
    trajs = [np.load(f"{path}/{f}") for f in files]
    return torch.from_numpy(np.stack(trajs)).float()


def main():
    device = torch.device(cfg.device)
    ckpt = torch.load("checkpoints/meta_epoch_50.pt", map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim

    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    encoder.eval(); sde.eval()

    source_scaler = ckpt.get('source_scaler', None)
    gen = torch.Generator(device=device).manual_seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(len(TASK_PAIRS), N_INTERP, figsize=(3 * N_INTERP, 3 * len(TASK_PAIRS)))
    if len(TASK_PAIRS) == 1:
        axes = axes[np.newaxis, :]

    alphas = np.linspace(0, 1, N_INTERP)
    rows = []

    for row_idx, (task_a, task_b) in enumerate(TASK_PAIRS):
        sup_a = load_task(task_a).to(device)
        sup_b = load_task(task_b).to(device)

        if source_scaler:
            sup_a_n = apply_scaler_to_trajectories(sup_a, source_scaler)
            sup_b_n = apply_scaler_to_trajectories(sup_b, source_scaler)
        else:
            sup_a_n, sup_b_n = sup_a, sup_b

        with torch.no_grad():
            z_a = encoder(sup_a_n[:, :50]).mean(dim=0)
            z_b = encoder(sup_b_n[:, :50]).mean(dim=0)

        T_full = cfg.time_grid.T
        n_steps = cfg.time_grid.n_steps
        x_max = cfg.stability.max_state_abs

        for col_idx, alpha in enumerate(alphas):
            z_interp = slerp(z_a, z_b, alpha).unsqueeze(0)
            x0 = sup_a_n[0:1, 0, :]

            with torch.no_grad():
                traj = simulate_neural_sde_batch(sde, x0, z_interp, T_full, n_steps, x_max, gen)

            traj_np = traj[0].cpu().numpy()
            ax = axes[row_idx, col_idx]
            ax.plot(traj_np[:, 0], traj_np[:, 1] if x_dim > 1 else traj_np[:, 0], alpha=0.8)
            ax.set_title(f"α={alpha:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            rows.append({'pair': f"{task_a}->{task_b}", 'alpha': alpha,
                         'mse_to_a': float(((traj[0, :sup_a_n.shape[1]] - sup_a_n[0]).pow(2)).mean()),
                         'mse_to_b': float(((traj[0, :sup_b_n.shape[1]] - sup_b_n[0]).pow(2)).mean())})

        axes[row_idx, 0].set_ylabel(f"{task_a}\n→\n{task_b}", fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/latent_interpolation.png", dpi=150)
    print("✅ Saved plots/latent_interpolation.png")

    import pandas as pd
    pd.DataFrame(rows).to_csv("results/latent_interpolation.csv", index=False)
    print("✅ Saved results/latent_interpolation.csv")


if __name__ == '__main__':
    main()
