# baselines/adapt_scratch.py
import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# --- CONFIG ---
# NAIVE REGRESSOR BASELINE (WEAKENED)
ADAPT_STEPS = 50       
ADAPT_LR = 3e-4       # Conservative LR
N_SHOTS = 2           
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
MC_SAMPLES = 5        # For NLL estimation

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def train_from_scratch_per_task(x_dim, z_dim, support_trajs, config, gen, limit_steps):
    """
    Trains a fresh SDE + Head from random initialization on the support set.
    """
    device = support_trajs.device
    start_time = time.time()
    
    # 1. Initialize Fresh Model (Low Capacity)
    sde = NeuralSDE(x_dim, z_dim, hidden_dim=32).to(device)
    head = ForecastHead(x_dim, z_dim, hidden_dim=32).to(device)

    # 2. Calculate Horizon
    full_T = config.time_grid.T
    full_steps = config.time_grid.n_steps
    dt = full_T / full_steps

    n_sim = limit_steps - 1
    T_train = dt * n_sim
    
    # Slice data
    train_data = support_trajs[:, :limit_steps, :] 
    x0 = train_data[:, 0, :]
    target_final = train_data[:, -1, :]
    
    x_max = config.stability.max_state_abs
    
    # 3. NOISY Z-INIT
    z_init = torch.randn(x0.size(0), z_dim, device=device) * 0.1
    z_init.requires_grad_(True)

    # 4. Optimizer
    optimizer = optim.Adam(
        list(sde.parameters()) + list(head.parameters()) + [z_init], 
        lr=ADAPT_LR, 
        weight_decay=1e-2 
    )

    sde.train(); head.train()

    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        
        # Simulate
        traj_pred = simulate_neural_sde_batch(
            sde, x0, z_init, T_train, n_sim, x_max, gen
        )

        pred_final = head(traj_pred[:, -1, :], z_init)

        # Path loss + head loss (mirrors Model C's adaptation objective).
        # FIX (fairness): the original code used head-only loss (MSE to the
        # final state only), while Model C uses full-trajectory path loss +
        # head loss + regularization.  Since the scratch SDE is trainable,
        # path loss provides gradient signal from every intermediate timestep,
        # closing the supervision gap.  Weak Transfer is exempt because its
        # SDE is frozen (path loss carries no gradient there).
        loss_path = F.mse_loss(traj_pred, train_data)
        loss_head = F.mse_loss(pred_final, target_final)
        loss = loss_path + loss_head
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(sde.parameters()) + list(head.parameters()) + [z_init], 1.0
        )
        optimizer.step()

    adapt_time = time.time() - start_time
    return sde, head, adapt_time

def evaluate_scratch_model(sde, head, query_trajs, config, gen):
    """
    Evaluates the task-specific model using MC sampling for NLL.
    """
    sde.eval()
    head.eval()
    device = query_trajs.device
    
    T = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs
    z_dim = config.latent.latent_dim

    x0_q = query_trajs[:, 0, :]
    # Use z=0 for query because task info is burnt into weights
    z_expanded = torch.zeros(x0_q.size(0), z_dim, device=device)

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            traj_q = simulate_neural_sde_batch(
                sde, x0_q, z_expanded, T, n_steps, x_max, gen
            )
            mc_preds.append(traj_q)
    
    # Statistics
    mc_tensor = torch.stack(mc_preds, dim=0)
    pred_mean = mc_tensor.mean(dim=0)
    pred_var = mc_tensor.var(dim=0) + 1e-6
    
    mse_rollout = F.mse_loss(pred_mean, query_trajs).item()
    mse_final = F.mse_loss(pred_mean[:, -1], query_trajs[:, -1, :]).item()
    nll = F.gaussian_nll_loss(pred_mean, query_trajs, pred_var).item()

    return mse_rollout, mse_final, nll

def main():
    device = torch.device(cfg.device)
    print("\n" + "=" * 80)
    print("💪 Scratch Baseline: FULL METRICS (Time, NLL, Rollout)")
    print("=" * 80)

    gen = torch.Generator(device=device)
    gen.manual_seed(999)

    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    out_file = "results/scratch_sweep_results_full.csv"
    
    # Resume Logic
    completed_keys = set()
    if os.path.exists(out_file):
        print(f"Found existing results. Resuming...")
        try:
            df_exist = pd.read_csv(out_file)
            for _, row in df_exist.iterrows():
                key = f"{row['regime']}_{row['theta_id']}_{row['steps_available']}"
                completed_keys.add(key)
        except:
            print("Error reading CSV, starting fresh.")

    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    for regime in ["testA", "testB", "testC"]:
        print(f"\nProcessing {regime}")
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        
        for theta_id in tqdm(tasks, desc=regime):
            
            # Optimization: Load data only if needed
            needed = False
            for steps in STEPS_SWEEP:
                if f"{regime}_{theta_id}_{steps}" not in completed_keys:
                    needed = True; break
            if not needed: continue

            full_support = get_task_data(ds_supp, theta_id, device)
            full_query = get_task_data(ds_query, theta_id, device)
            
            task_results = []
            
            for limit_steps in STEPS_SWEEP:
                key = f"{regime}_{theta_id}_{limit_steps}"
                if key in completed_keys: continue

                # Train
                sde_s, head_s, adapt_time = train_from_scratch_per_task(
                    x_dim, z_dim, full_support[:N_SHOTS], cfg, gen, limit_steps
                )
                
                # Eval
                mse_rollout, mse_final, nll = evaluate_scratch_model(
                    sde_s, head_s, full_query, cfg, gen
                )
                
                task_results.append({
                    "regime": regime, 
                    "theta_id": theta_id,
                    "steps_available": limit_steps,
                    "mse_rollout": mse_rollout,
                    "mse_final": mse_final,
                    "nll": nll,
                    "adapt_time": adapt_time
                })
            
            if task_results:
                df_task = pd.DataFrame(task_results)
                mode = 'a' if os.path.exists(out_file) else 'w'
                header = not os.path.exists(out_file)
                df_task.to_csv(out_file, mode=mode, header=header, index=False)

    print(f"\n✅ Baseline Sweep Complete → {out_file}")
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        # Show full columns
        pd.set_option('display.max_rows', None)
        print(df.groupby(["regime", "steps_available"])[["mse_rollout", "mse_final", "nll", "adapt_time"]].mean())

if __name__ == "__main__":
    main()