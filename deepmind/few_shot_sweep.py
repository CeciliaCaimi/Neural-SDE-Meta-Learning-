import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from dataloaders.trajectory_datasets import TrajectoryDataset

# --- SETTINGS ---
CKPT_PATH = "checkpoints/transfer_epoch_50.pt"
DATA_ROOT = "data"
INDEX_PATH = os.path.join(DATA_ROOT, "index.csv")
X_DIM = 10
HIDDEN_DIM = 128
N_SHOTS = [1, 2, 5, 10]
TASKS_TO_EVAL = 20  # Average over this many tasks for a smooth curve

def get_smart_split(index_path):
    if not os.path.exists(index_path): return None, None
    df = pd.read_csv(index_path)
    available = df['split'].unique()
    print(f"   ℹ️  Splits found: {available}")
    for cand in ['val', 'testA', 'test', 'train']:
        if cand in available:
            roles = df[df['split'] == cand]['role'].unique()
            target_role = 'query' if 'query' in roles else roles[0]
            return cand, target_role
    return available[0], df[df['split']==available[0]]['role'].unique()[0]

def run_sweep():
    print(f"🧪 STARTING ROBUST FEW-SHOT SWEEP (Averaging {TASKS_TO_EVAL} tasks)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    encoder = TrajEncoder(X_DIM, cfg.latent.latent_dim, HIDDEN_DIM).to(device)
    sde = NeuralSDE(X_DIM, cfg.latent.latent_dim, HIDDEN_DIM).to(device)
    
    ckpt = torch.load(CKPT_PATH, map_location=device)
    enc_key = 'encoder_state_dict' if 'encoder_state_dict' in ckpt else 'encoder'
    sde_key = 'sde_state_dict' if 'sde_state_dict' in ckpt else 'sde'
    encoder.load_state_dict(ckpt[enc_key]); sde.load_state_dict(ckpt[sde_key])
    encoder.eval(); sde.eval()
    
    # 2. Load Data
    split, role = get_smart_split(INDEX_PATH)
    if not split: return
    ds = TrajectoryDataset(INDEX_PATH, split, role)
    
    # Get list of tasks that have enough data
    all_tasks = ds.metadata["theta_id"].unique()
    valid_tasks = []
    for t in all_tasks:
        if len(ds.metadata[ds.metadata["theta_id"] == t]) >= 11: # Need 10 context + 1 target
            valid_tasks.append(t)
            
    print(f"   ✅ Found {len(valid_tasks)} valid tasks. Evaluating on first {min(len(valid_tasks), TASKS_TO_EVAL)}...")
    
    # 3. Run Sweep with Progress Bar
    results = {n: [] for n in N_SHOTS}
    
    # Loop over tasks
    for task_id in tqdm(valid_tasks[:TASKS_TO_EVAL], desc="Tasks"):
        rows = ds.metadata[ds.metadata["theta_id"] == task_id]
        indices = rows.index.tolist()
        
        # We always use the LAST trajectory as the target (held-out)
        target_idx = indices[-1]
        target_traj = ds[target_idx][0].to(device)
        if target_traj.shape[-1] > X_DIM: target_traj = target_traj[:, :X_DIM]
        
        # Loop over shot counts
        for n in N_SHOTS:
            context_indices = indices[:n]
            
            # Prepare Batch
            ctx_list = []
            for idx in context_indices:
                traj = ds[idx][0].to(device)
                if traj.shape[-1] > X_DIM: traj = traj[:, :X_DIM]
                ctx_list.append(traj[:20]) # Context window
                
            ctx_batch = torch.stack(ctx_list)
            
            with torch.no_grad():
                # Inference
                z_ind = encoder(ctx_batch)
                if isinstance(z_ind, tuple): z_ind = z_ind[0]
                z_final = torch.mean(z_ind, dim=0, keepdim=True)
                
                # Predict
                x_curr = target_traj[:-1]
                z_exp = z_final.expand(x_curr.size(0), -1)
                drift = sde.f(0, x_curr, z_exp)
                x_pred = x_curr + drift * 0.05
                
                mse = F.mse_loss(x_pred, target_traj[1:]).item()
                results[n].append(mse)

    # 4. Aggregate & Plot
    final_means = []
    final_stds = []
    print("\n📊 FINAL RESULTS (Averaged):")
    print("-" * 30)
    for n in N_SHOTS:
        avg_mse = np.mean(results[n])
        std_mse = np.std(results[n])
        final_means.append(avg_mse)
        final_stds.append(std_mse)
        print(f"   N={n:<2} | Mean MSE: {avg_mse:.6f} (±{std_mse:.6f})")
        
    df = pd.DataFrame({"n_shots": N_SHOTS, "mse_mean": final_means, "mse_std": final_stds})
    df.to_csv("results/few_shot_scaling.csv", index=False)
    
    plt.figure(figsize=(7, 5))
    plt.errorbar(N_SHOTS, final_means, yerr=final_stds, fmt='-o', color='purple', ecolor='gray', capsize=5)
    plt.xlabel("Number of Context Trajectories (N)")
    plt.ylabel("Test MSE")
    plt.title(f"Few-Shot Adaptation (Avg over {TASKS_TO_EVAL} Tasks)")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/plots/few_shot_scaling.png")
    print(f"\n✅ Robust plot saved to results/plots/few_shot_scaling.png")

if __name__ == "__main__":
    run_sweep()