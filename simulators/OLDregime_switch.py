#simulators/regime_switch.py 

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch
from dataloaders.trajectory_datasets import TrajectoryDataset

def get_task_params(dataset, theta_id):
    # Helper to get the raw data for a task
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    # Return one trajectory to act as "Ground Truth" physics
    return dataset[idx[0]][0]

def run_regime_switch():
    print("🔄 Running Regime Switching Experiment (Sudden Physics Change)...")
    device = torch.device(cfg.device)
    
    # 1. Load Pre-trained Meta-Model
    # (Assuming you have a good checkpoint from your main training)
    # If not, use the latest one you have.
    ckpt_path = "checkpoints/meta_epoch_50.pt" 
    if not os.path.exists(ckpt_path):
        print("⚠️ Checkpoint not found, skipping.")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    
    encoder.load_state_dict(ckpt['encoder']) # Check keys! might be 'encoder_state_dict'
    sde.load_state_dict(ckpt['sde'])
    
    encoder.eval()
    sde.eval()
    
    # 2. Setup "Frankenstein" Trajectory
    # We take Task A (Normal) and Task C (Chaos)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    ds_A = TrajectoryDataset(index_path, "testA", "query")
    ds_C = TrajectoryDataset(index_path, "testC", "query")
    
    # Pick two random tasks
    task_A_id = ds_A.metadata["theta_id"].unique()[0]
    task_C_id = ds_C.metadata["theta_id"].unique()[0]
    
    traj_A = get_task_params(ds_A, task_A_id).to(device)
    traj_C = get_task_params(ds_C, task_C_id).to(device)
    
    # Create Switch: 0-50 steps is A, 51-100 steps is C
    # We artificially stitch them. Note: This creates a "Jump" in state space.
    # A real physical switch would keep state continuous but change physics f(x).
    # Ideally, we simulate: dx = f_A(x) for t<T, dx = f_C(x) for t>T.
    # Since we can't easily re-simulate ground truth without the generator code, 
    # we will rely on the pre-generated trajectories and assume a "teleport" or "hard shock".
    
    T_switch = 50
    ground_truth = torch.cat([traj_A[:T_switch], traj_C[:T_switch]], dim=0) # Total 100 steps
    
    # 3. Online Adaptation Loop
    # We use a sliding window context of 20 steps to infer z
    
    z_history = []
    reconstructions = []
    errors = []
    
    window_size = 20
    
    print(f"Simulating Switch: Task {task_A_id} -> Task {task_C_id} at step {T_switch}")
    
    for t in range(window_size, len(ground_truth)):
        # Sliding Window Context
        context = ground_truth[t-window_size : t].unsqueeze(0) # (1, 20, D)
        
        with torch.no_grad():
            # Infer Z from recent history
            z = encoder(context).mean(0)
            z_history.append(z.cpu().numpy())
            
            # Predict Next Step (One-step ahead)
            # We use the SDE to predict t -> t+1
            x_curr = ground_truth[t-1].unsqueeze(0)
            # Simulate 1 step
            # Note: simulate_neural_sde_batch expects batch
            z_exp = z.unsqueeze(0)
            
            # Manual Euler Step for speed/simplicity
            # dt = 0.005
            drift = sde.f(0, x_curr, z_exp)
            x_next_pred = x_curr + drift * 0.005
            
            reconstructions.append(x_next_pred.squeeze().cpu().numpy())
            
            # Error
            x_true = ground_truth[t].unsqueeze(0)
            mse = F.mse_loss(x_next_pred, x_true).item()
            errors.append(mse)

    # 4. Save & Plot
    os.makedirs("results", exist_ok=True)
    
    # Plot Error over Time
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="One-Step MSE")
    plt.axvline(x=T_switch - window_size, color='r', linestyle='--', label="Physics Switch")
    plt.title("Online Adaptation: Error Spike & Recovery")
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("results/regime_switch_error.png")
    
    # Save Data
    df = pd.DataFrame({
        "time": range(window_size, len(ground_truth)),
        "error": errors,
        "regime": ["A"] * (T_switch - window_size) + ["C"] * (len(errors) - (T_switch - window_size))
    })
    df.to_csv("results/regime_switch_data.csv", index=False)
    print("✅ Regime Switch Experiment Done. Plot saved to results/regime_switch_error.png")

if __name__ == "__main__":
    run_regime_switch()