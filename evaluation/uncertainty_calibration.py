"""Lightweight Uncertainty Calibration Analysis.

Computes variance-error correlation and reliability curve using
MC dropout samples from the SDE rollout.

For each task:
  - Run MC_SAMPLES rollouts (already stochastic via SDE diffusion)
  - Compute per-timestep predictive variance
  - Compute per-timestep squared error
  - Report Pearson correlation(variance, error)
  - Bin predictions by variance quantile, plot mean error per bin (reliability)

Outputs:
    results/uncertainty_calibration.csv
    plots/uncertainty_reliability.png
"""
import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
sys.path.append(os.getcwd())

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import (
    adapt_model, simulate_neural_sde_batch, MC_SAMPLES
)
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories, apply_scaler_to_trajectories

N_TASKS = 15
REGIMES = ["testA", "testB", "testC"]
N_MC = 20  # more samples for better variance estimate
N_BINS = 10


def main():
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(42)

    ckpt = torch.load("checkpoints/meta_epoch_50.pt", map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim

    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])
    encoder.eval(); sde.eval(); head.eval()

    source_scaler = ckpt.get('source_scaler', None)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    all_variances = []
    all_errors = []
    rows = []

    for regime in REGIMES:
        data_dir = "data/test_trajectories"
        task_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith(regime)])[:N_TASKS]

        for task_id in task_dirs:
            sup_path = f"{data_dir}/{task_id}/support"
            qry_path = f"{data_dir}/{task_id}/query"
            if not os.path.exists(sup_path) or not os.path.exists(qry_path):
                continue

            sup_files = sorted(os.listdir(sup_path))[:5]
            qry_files = sorted(os.listdir(qry_path))[:5]
            support = torch.from_numpy(np.stack([np.load(f"{sup_path}/{f}") for f in sup_files])).float().to(device)
            query = torch.from_numpy(np.stack([np.load(f"{qry_path}/{f}") for f in qry_files])).float().to(device)

            target_scaler = fit_scaler_on_trajectories(support) if source_scaler else None
            support_n = apply_scaler_to_trajectories(support, target_scaler) if target_scaler else support
            query_n = apply_scaler_to_trajectories(query, target_scaler) if target_scaler else query

            # Encode + adapt
            with torch.no_grad():
                z_init = encoder(support_n[:, :50]).mean(dim=0, keepdim=True)
            head_opt, z_opt, _ = adapt_model(sde, head, z_init, support_n, gen, cfg)

            # MC rollouts
            B_q = query_n.shape[0]
            z_batch = z_opt.expand(B_q, -1)
            T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
            x_max = cfg.stability.max_state_abs

            mc_trajs = []
            with torch.no_grad():
                for _ in range(N_MC):
                    traj = simulate_neural_sde_batch(sde, query_n[:, 0], z_batch, T_full, n_steps, x_max, gen)
                    mc_trajs.append(traj)

            mc_stack = torch.stack(mc_trajs)  # (N_MC, B_q, T, D)
            valid_len = min(mc_stack.shape[2], query_n.shape[1])

            pred_mean = mc_stack[:, :, :valid_len].mean(0)
            pred_var = mc_stack[:, :, :valid_len].var(0).mean(-1)  # (B_q, T) avg over dims
            sq_error = (pred_mean - query_n[:, :valid_len]).pow(2).mean(-1)  # (B_q, T)

            var_flat = pred_var.cpu().numpy().flatten()
            err_flat = sq_error.cpu().numpy().flatten()

            corr, pval = pearsonr(var_flat, err_flat)
            rows.append({'regime': regime, 'theta_id': task_id,
                         'pearson_r': corr, 'p_value': pval,
                         'mean_var': var_flat.mean(), 'mean_error': err_flat.mean()})

            all_variances.extend(var_flat.tolist())
            all_errors.extend(err_flat.tolist())

            print(f"  {task_id}: r={corr:.3f} (p={pval:.2e})")

    df = pd.DataFrame(rows)
    df.to_csv("results/uncertainty_calibration.csv", index=False)

    # Reliability curve
    all_var = np.array(all_variances)
    all_err = np.array(all_errors)
    quantiles = np.quantile(all_var, np.linspace(0, 1, N_BINS + 1))

    bin_means_var = []
    bin_means_err = []
    for i in range(N_BINS):
        mask = (all_var >= quantiles[i]) & (all_var < quantiles[i + 1])
        if mask.sum() > 0:
            bin_means_var.append(all_var[mask].mean())
            bin_means_err.append(all_err[mask].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Reliability curve
    ax1.plot(bin_means_var, bin_means_err, 'o-', color='steelblue')
    max_val = max(max(bin_means_var), max(bin_means_err))
    ax1.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, label='Perfect calibration')
    ax1.set_xlabel("Predictive Variance (binned)")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Reliability Curve")
    ax1.legend()

    # Correlation summary by regime
    for regime in REGIMES:
        sub = df[df['regime'] == regime]
        ax2.bar(regime, sub['pearson_r'].mean(), yerr=sub['pearson_r'].std(), capsize=5, alpha=0.7)
    ax2.set_ylabel("Pearson r (variance vs error)")
    ax2.set_title("Variance-Error Correlation by Regime")
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/uncertainty_reliability.png", dpi=150)
    print(f"\n✅ Saved plots/uncertainty_reliability.png")

    # Summary
    print(f"\n=== Uncertainty Calibration Summary ===")
    print(f"{'Regime':<8} {'Pearson r':<12} {'p-value':<12}")
    for regime in REGIMES:
        sub = df[df['regime'] == regime]
        print(f"{regime:<8} {sub['pearson_r'].mean():<12.3f} {sub['p_value'].mean():<12.2e}")
    print(f"\n✅ Saved results/uncertainty_calibration.csv")


if __name__ == '__main__':
    main()
