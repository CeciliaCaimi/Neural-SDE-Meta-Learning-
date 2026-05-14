"""Latent Dimensionality Sweep (fixed).

Evaluates z ∈ {4, 16, 64} using checkpoints that were actually trained
at those z_dims. Reads z_dim from the checkpoint's fc.weight shape.

Outputs:
    results/zdim_sweep.csv
    plots/zdim_sweep.png
"""
import os, sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import gated_inference
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories

CHECKPOINT_MAP = {
    4:  "checkpoints/meta_zdim4_epoch_50.pt",
    16: "checkpoints/meta_epoch_50.pt",
    64: "checkpoints/meta_zdim64_epoch_50.pt",
}
N_TASKS = 10
REGIMES = ["testA", "testB", "testC"]


def evaluate_zdim(z_dim_label, device, gen):
    ckpt_path = CHECKPOINT_MAP.get(z_dim_label)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"  ⚠️  z_dim={z_dim_label}: no checkpoint ({ckpt_path}), skipping")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Read actual z_dim from checkpoint
    z_dim = ckpt['encoder']['fc.weight'].shape[0]
    print(f"  Checkpoint {ckpt_path}: actual z_dim={z_dim}")

    x_dim = cfg.basis.x_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])
    encoder.eval(); sde.eval(); head.eval()

    source_scaler = ckpt.get('source_scaler', None)

    # Temporarily override cfg for gated_inference
    orig_zdim = cfg.latent.latent_dim
    cfg.latent.latent_dim = z_dim
    results = []

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

            try:
                metrics = gated_inference(encoder, sde, head, support, query, gen, cfg, target_scaler)
                results.append({'z_dim': z_dim, 'regime': regime, 'theta_id': task_id,
                                'mse_rollout': metrics['mse_rollout']})
            except Exception as e:
                print(f"    {task_id} failed: {e}")

    cfg.latent.latent_dim = orig_zdim
    return results


def main():
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(42)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    all_results = []
    for z_dim in sorted(CHECKPOINT_MAP.keys()):
        print(f"\n--- z_dim = {z_dim} ---")
        res = evaluate_zdim(z_dim, device, gen)
        if res:
            all_results.extend(res)
            mean_mse = np.mean([r['mse_rollout'] for r in res])
            print(f"  Mean MSE: {mean_mse:.4f} ({len(res)} tasks)")

    if not all_results:
        print("No results. Need to train checkpoints first.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv("results/zdim_sweep.csv", index=False)
    print(f"\n✅ Saved results/zdim_sweep.csv")

    # Plot
    summary = df.groupby(['z_dim', 'regime'])['mse_rollout'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime in REGIMES:
        sub = summary[summary['regime'] == regime]
        ax.errorbar(sub['z_dim'], sub['mean'], yerr=sub['std'], marker='o', label=regime, capsize=3)
    ax.set_xlabel("Latent Dimension (z_dim)")
    ax.set_ylabel("MSE Rollout")
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(CHECKPOINT_MAP.keys()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.set_title("Latent Dimensionality Sweep")
    plt.tight_layout()
    plt.savefig("plots/zdim_sweep.png", dpi=150)
    print("✅ Saved plots/zdim_sweep.png")


if __name__ == '__main__':
    main()
