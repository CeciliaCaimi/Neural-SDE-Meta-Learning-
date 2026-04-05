import os
import copy
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# --- CONFIG ---
# Varying the "N" in Few-Shot
SHOTS_SWEEP = [2, 3, 4, 5, 8, 10]

# FAIR COMPARISON: Both get exactly the same budget
ADAPT_STEPS = 100   
LR_HEAD = 1e-2       # Fast adaptation for Head
LR_SCRATCH = 2e-3    # Standard LR for Scratch

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def run_meta_sde(encoder_base, sde_base, head_base, support, query, config, gen):
    # 1. Zero-shot Inference
    encoder_base.eval()
    with torch.no_grad():
        z_all = encoder_base(support) 
        z_mean = z_all.mean(dim=0, keepdim=True)

    # 2. Adaptation (Head Only)
    head_ft = copy.deepcopy(head_base)
    head_ft.train()
    optimizer = optim.Adam(head_ft.parameters(), lr=LR_HEAD)
    sde_base.eval() # SDE Frozen
    
    T_full = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs
    z_expanded = z_mean.expand(support.size(0), -1)
    
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        traj = simulate_neural_sde_batch(sde_base, support[:,0,:], z_expanded, T_full, n_steps, x_max, gen)
        pred = head_ft(traj[:, -1, :], z_expanded)
        loss = F.mse_loss(pred, support[:, -1, :])
        loss.backward()
        optimizer.step()
        
    # 3. Evaluation
    head_ft.eval()
    with torch.no_grad():
        z_q = z_mean.expand(query.size(0), -1)
        traj_q = simulate_neural_sde_batch(sde_base, query[:,0,:], z_q, T_full, n_steps, x_max, gen)
        pred_q = head_ft(traj_q[:, -1, :], z_q)
        mse = F.mse_loss(pred_q, query[:, -1, :]).item()
        
    return mse

def run_scratch_adaptation(x_dim, z_dim, support, query, config, gen):
    # Initialize Randomly (Scratch)
    sde = NeuralSDE(x_dim, z_dim, 64).to(support.device)
    head = ForecastHead(x_dim, z_dim, 64).to(support.device)
    optimizer = optim.Adam(list(sde.parameters()) + list(head.parameters()), lr=LR_SCRATCH)
    
    T_full = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs
    z_zeros = torch.zeros(support.size(0), z_dim, device=support.device)

    # Adapt (Train)
    sde.train()
    head.train()
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        traj = simulate_neural_sde_batch(sde, support[:,0,:], z_zeros, T_full, n_steps, x_max, gen)
        pred = head(traj[:, -1, :], z_zeros)
        loss = F.mse_loss(pred, support[:, -1, :]) + F.mse_loss(traj, support)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sde.parameters(), 1.0)
        optimizer.step()
        
    # Eval
    sde.eval()
    head.eval()
    with torch.no_grad():
        z_q = torch.zeros(query.size(0), z_dim, device=support.device)
        traj_q = simulate_neural_sde_batch(sde, query[:,0,:], z_q, T_full, n_steps, x_max, gen)
        pred_q = head(traj_q[:, -1, :], z_q)
        mse = F.mse_loss(pred_q, query[:, -1, :]).item()
        
    return mse

def main():
    device = torch.device(cfg.device)
    print("🎯 Running SHOT SWEEP (Fair Comparison)")
    print(f"Shots: {SHOTS_SWEEP} | Adapt Steps: {ADAPT_STEPS}")
    
    # Load Meta-Model
    ckpt = torch.load("checkpoints/meta_epoch_50.pt", map_location=device)
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    
    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])
    
    gen = torch.Generator(device=device)
    gen.manual_seed(999)
    
    out_file = "results/shot_sweep_results.csv"
    file_exists = os.path.isfile(out_file)
    
    # Data Setup
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    ds_supp = TrajectoryDataset(index_path, "testC", "support", check_shapes=True)
    ds_query = TrajectoryDataset(index_path, "testC", "query", check_shapes=True)
    
    tasks = ds_supp.metadata["theta_id"].unique()
    
    # --- LOOP ---
    for theta_id in tqdm(tasks, desc="Test C"):
        full_support = get_task_data(ds_supp, theta_id, device)
        full_query = get_task_data(ds_query, theta_id, device)
        max_shots = full_support.shape[0]
        
        task_results = []
        
        for n_shots in SHOTS_SWEEP:
            if n_shots > max_shots: continue
            
            support_set = full_support[:n_shots]
            
            # Run Both
            mse_meta = run_meta_sde(encoder, sde, head, support_set, full_query, cfg, gen)
            mse_scratch = run_scratch_adaptation(x_dim, z_dim, support_set, full_query, cfg, gen)
            
            task_results.append({
                "theta_id": theta_id,
                "n_shots": n_shots,
                "MSE_Meta": mse_meta,
                "MSE_Scratch": mse_scratch
            })
            
        # SAVE AFTER EVERY TASK (Safety)
        df_task = pd.DataFrame(task_results)
        df_task.to_csv(out_file, mode='a', header=not file_exists, index=False)
        file_exists = True

    print(f"\n✅ Done. Saved to {out_file}")

if __name__ == "__main__":
    main()