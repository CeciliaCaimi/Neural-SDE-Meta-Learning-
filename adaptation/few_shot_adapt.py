# adaptation/few_shot_adapt.py
# HERO RUN: Data Efficiency Sweep (Granular)
# Measures performance as a function of available support steps.

import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# --- CONFIG ---
N_SHOTS = 2

# Adaptation: only encoder + head, SDE frozen
ADAPT_STEPS = 50
LR_HEAD = 1e-2

# THE GRANULAR SWEEP
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    indices = rows.index.tolist()
    data = [dataset[i][0] for i in indices]
    return torch.stack(data).to(device)

def infer_z_star(encoder, support_trajs, obs_len):
    """Infer z from the OBSERVED PAST only."""
    encoder.eval()
    with torch.no_grad():
        obs = support_trajs[:, :obs_len, :]
        z_all = encoder(obs)
        z_mean = z_all.mean(dim=0, keepdim=True)
    return z_mean

def evaluate_model(sde, head, z, query_trajs, config, gen):
    """Evaluate on Query set (Full Future 0->T)."""
    sde.eval()
    head.eval()
    T = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs

    with torch.no_grad():
        x0 = query_trajs[:, 0, :]
        z_expanded = z.expand(x0.size(0), -1)

        # Always simulate full duration for evaluation
        traj_pred = simulate_neural_sde_batch(
            sde, x0, z_expanded, T, n_steps, x_max, gen
        )
        x_T_pred = traj_pred[:, -1, :]
        final_pred = head(x_T_pred, z_expanded)

        mse_path = torch.mean((traj_pred - query_trajs) ** 2).item()
        mse_head = torch.mean((final_pred - query_trajs[:, -1, :]) ** 2).item()

    return mse_path, mse_head

def fine_tune_frozen_sde(encoder, sde, head, support_trajs, config, gen, limit_steps):
    """
    Few-shot adaptation on LIMITED history:
    - Slices support_trajs to [:limit_steps]
    - Freezes SDE
    - Adapts encoder + head using *final-step* MSE only
    """
    encoder_ft = copy.deepcopy(encoder)
    sde_ft = copy.deepcopy(sde)
    head_ft = copy.deepcopy(head)
    
    # 1. Prepare Training Data (Slice based on availability)
    train_target = support_trajs[:, :limit_steps, :]
    x0 = train_target[:, 0, :]
    target_final = train_target[:, -1, :]
    
    # 2. Calculate Simulation Params for this partial horizon
    T_full = config.time_grid.T
    n_steps_full = config.time_grid.n_steps
    dt = T_full / n_steps_full
    
    # limit_steps indices 0..limit_steps-1 --> limit_steps-1 steps to reach that time
    n_sim = limit_steps - 1
    T_train = dt * n_sim
    x_max = config.stability.max_state_abs
    
    # Encoder sees up to 50 steps max, or limit_steps if smaller
    enc_obs_len = min(limit_steps, 50)

    # Optimizer: encoder + head only
    param_group = list(encoder_ft.parameters()) + list(head_ft.parameters())
    optimizer = optim.Adam(param_group, lr=LR_HEAD)
    
    # Freeze SDE
    sde_ft.eval()
    for p in sde_ft.parameters():
        p.requires_grad = False

    encoder_ft.train()
    head_ft.train()
    
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        
        # Infer z from observed prefix
        obs = train_target[:, :enc_obs_len, :]
        z = encoder_ft(obs).mean(dim=0, keepdim=True)
        z_expanded = z.expand(x0.size(0), -1)
        
        # Simulate partial path (SDE frozen)
        traj_pred = simulate_neural_sde_batch(
            sde_ft, x0, z_expanded, T_train, n_sim, x_max, gen
        )
        
        x_T = traj_pred[:, -1, :]
        pred = head_ft(x_T, z_expanded)
        
        # Final-step MSE only
        loss = nn.functional.mse_loss(pred, target_final)
        
        loss.backward()
        optimizer.step()
        
    return sde_ft, head_ft, z

def main():
    device = torch.device(cfg.device)
    print("🔬 Meta-Learning: Data Efficiency Sweep")
    print(f"Sweeping Step Limits: {STEPS_SWEEP}")
    print(f"Adaptation: {ADAPT_STEPS} steps (encoder + head), SDE frozen")

    ckpt_path = "checkpoints/meta_epoch_50.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    # Load Base Models
    encoder_base = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde_base = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head_base = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    encoder_base.load_state_dict(ckpt["encoder"])
    sde_base.load_state_dict(ckpt["sde"])
    head_base.load_state_dict(ckpt["head"])

    gen = torch.Generator(device=device)
    gen.manual_seed(999)

    results_list = []
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    # Regimes Loop
    for regime in ["testA", "testB", "testC"]:
        print(f"\n🎯 Processing {regime}")
        try:
            ds_support = TrajectoryDataset(index_path, regime, "support", check_shapes=True)
            ds_query = TrajectoryDataset(index_path, regime, "query", check_shapes=True)
        except:
            continue

        # Tasks Loop
        tasks = ds_support.metadata["theta_id"].unique()
        for theta_id in tqdm(tasks, desc=regime):
            full_support = get_task_data(ds_support, theta_id, device)
            full_query = get_task_data(ds_query, theta_id, device)
            
            # --- THE SWEEP ---
            for limit_steps in STEPS_SWEEP:
                # 1. Zero-shot (uses only limit_steps for context)
                enc_obs_len = min(limit_steps, 50)
                z_star = infer_z_star(encoder_base, full_support[:N_SHOTS], enc_obs_len)
                zs_mse_path, zs_mse_head = evaluate_model(
                    sde_base, head_base, z_star, full_query, cfg, gen
                )

                # 2. Few-shot: encoder + head adaptation, SDE frozen
                sde_ft, head_ft, z_ft = fine_tune_frozen_sde(
                    encoder_base, sde_base, head_base, 
                    full_support[:N_SHOTS], cfg, gen, limit_steps
                )
                
                fs_mse_path, fs_mse_head = evaluate_model(
                    sde_ft, head_ft, z_ft, full_query, cfg, gen
                )

                results_list.append({
                    "regime": regime,
                    "theta_id": theta_id,
                    "steps_available": limit_steps,
                    "mse_head_zeroshot": zs_mse_head,
                    "mse_head_fewshot": fs_mse_head,
                    "mse_path_fewshot": fs_mse_path
                })

    # Save Results
    df = pd.DataFrame(results_list)
    out_path = "results/efficiency_sweep_results.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print("\n✅ Sweep Complete.")
    print("\n📊 Average Head MSE by Regime & Steps:")
    print(df.groupby(["regime", "steps_available"])[["mse_head_fewshot"]].mean())

if __name__ == "__main__":
    main()
