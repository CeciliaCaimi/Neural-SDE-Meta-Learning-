"""Compact Adaptation Comparison Table.

Compares four adaptation strategies:
  1. Encoder only (z_init from encoder, no gradient steps)
  2. Encoder + latent refinement (optimize z only, freeze head)
  3. Full adaptation (optimize z + head — default Model C)
  4. Scratch (random z_init, optimize z + head)

Outputs:
    results/adaptation_comparison.csv
    results/adaptation_comparison_table.txt
"""
import os, sys, copy, time, torch, numpy as np, pandas as pd
import torch.nn.functional as F
import torch.optim as optim
sys.path.append(os.getcwd())

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import (
    simulate_neural_sde_batch, compute_residual, gated_inference,
    ADAPT_STEPS, LR_HEAD, LR_Z, BETA_REG, MC_SAMPLES
)
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories, apply_scaler_to_trajectories

N_TASKS = 10
REGIMES = ["testA", "testB", "testC"]


def encoder_only(encoder, sde, head, support_n, query_n, gen, cfg_obj):
    """No adaptation — just use encoder output directly."""
    with torch.no_grad():
        z = encoder(support_n[:, :50]).mean(dim=0, keepdim=True)
        B_q = query_n.shape[0]
        z_batch = z.expand(B_q, -1)
        T_full = cfg_obj.time_grid.T; n_steps = cfg_obj.time_grid.n_steps
        x_max = cfg_obj.stability.max_state_abs
        preds = []
        for _ in range(MC_SAMPLES):
            traj = simulate_neural_sde_batch(sde, query_n[:, 0], z_batch, T_full, n_steps, x_max, gen)
            preds.append(traj)
        pred = torch.stack(preds).mean(0)
        valid = min(pred.shape[1], query_n.shape[1])
        mse = F.mse_loss(pred[:, :valid], query_n[:, :valid]).item()
    return mse


def encoder_plus_refine(encoder, sde, head_init, support_n, query_n, gen, cfg_obj):
    """Optimize z only, freeze head."""
    with torch.no_grad():
        z_init = encoder(support_n[:, :50]).mean(dim=0, keepdim=True)

    z = z_init.clone().detach(); z.requires_grad = True
    optimizer = optim.Adam([z], lr=LR_Z)

    B, T, D = support_n.shape
    T_full = cfg_obj.time_grid.T; n_steps = cfg_obj.time_grid.n_steps
    dt = T_full / n_steps; n_sim = T - 1; T_sim = dt * n_sim
    x_max = cfg_obj.stability.max_state_abs

    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        z_batch = z.expand(B, -1)
        traj = simulate_neural_sde_batch(sde, support_n[:, 0], z_batch, T_sim, n_sim, x_max, gen)
        valid = min(traj.shape[1], T)
        loss = F.mse_loss(traj[:, :valid], support_n[:, :valid]) + BETA_REG * z.pow(2).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        B_q = query_n.shape[0]
        z_batch = z.detach().expand(B_q, -1)
        preds = []
        for _ in range(MC_SAMPLES):
            t = simulate_neural_sde_batch(sde, query_n[:, 0], z_batch, T_full, n_steps, x_max, gen)
            preds.append(t)
        pred = torch.stack(preds).mean(0)
        valid = min(pred.shape[1], query_n.shape[1])
        mse = F.mse_loss(pred[:, :valid], query_n[:, :valid]).item()
    return mse


def scratch_adapt(sde, head_init, support_n, query_n, gen, cfg_obj, z_dim):
    """Random z_init (no encoder), optimize z + head."""
    z_init = torch.randn(1, z_dim, device=support_n.device) * 0.1
    z = z_init.clone().detach(); z.requires_grad = True
    head = copy.deepcopy(head_init); head.train()
    optimizer = optim.Adam([{'params': head.parameters(), 'lr': LR_HEAD}, {'params': [z], 'lr': LR_Z}])

    B, T, D = support_n.shape
    T_full = cfg_obj.time_grid.T; n_steps = cfg_obj.time_grid.n_steps
    dt = T_full / n_steps; n_sim = T - 1; T_sim = dt * n_sim
    x_max = cfg_obj.stability.max_state_abs

    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        z_batch = z.expand(B, -1)
        traj = simulate_neural_sde_batch(sde, support_n[:, 0], z_batch, T_sim, n_sim, x_max, gen)
        valid = min(traj.shape[1], T)
        loss = F.mse_loss(traj[:, :valid], support_n[:, :valid]) + BETA_REG * z.pow(2).sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        B_q = query_n.shape[0]
        z_batch = z.detach().expand(B_q, -1)
        preds = []
        for _ in range(MC_SAMPLES):
            t = simulate_neural_sde_batch(sde, query_n[:, 0], z_batch, T_full, n_steps, x_max, gen)
            preds.append(t)
        pred = torch.stack(preds).mean(0)
        valid = min(pred.shape[1], query_n.shape[1])
        mse = F.mse_loss(pred[:, :valid], query_n[:, :valid]).item()
    return mse


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

            # 1. Encoder only
            mse_enc = encoder_only(encoder, sde, head, support_n, query_n, gen, cfg)

            # 2. Encoder + refine z
            mse_refine = encoder_plus_refine(encoder, sde, head, support_n, query_n, gen, cfg)

            # 3. Full adaptation (default Model C)
            metrics = gated_inference(encoder, sde, head, support, query, gen, cfg, target_scaler)
            mse_full = metrics['mse_rollout']

            # 4. Scratch
            mse_scratch = scratch_adapt(sde, head, support_n, query_n, gen, cfg, z_dim)

            rows.append({'regime': regime, 'theta_id': task_id,
                         'encoder_only': mse_enc, 'encoder_refine': mse_refine,
                         'full_adaptation': mse_full, 'scratch': mse_scratch})
            print(f"  {task_id}: enc={mse_enc:.4f} ref={mse_refine:.4f} full={mse_full:.4f} scr={mse_scratch:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/adaptation_comparison.csv", index=False)

    # Print compact table
    print("\n\n=== Adaptation Comparison (Mean MSE) ===")
    print(f"{'Regime':<8} {'Encoder':<10} {'Enc+Refine':<12} {'Full':<10} {'Scratch':<10}")
    for regime in REGIMES:
        sub = df[df['regime'] == regime]
        print(f"{regime:<8} {sub['encoder_only'].mean():<10.4f} {sub['encoder_refine'].mean():<12.4f} "
              f"{sub['full_adaptation'].mean():<10.4f} {sub['scratch'].mean():<10.4f}")

    print(f"\n✅ Saved results/adaptation_comparison.csv")


if __name__ == '__main__':
    main()
