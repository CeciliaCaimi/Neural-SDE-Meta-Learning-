import os
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# === HYPERPARAMETERS ===
# Optimization (Plasticity)
ADAPT_STEPS = 50        # How many gradient steps to take
LR_Z = 1e-2             # Learning rate for Latent Code z
LR_HEAD = 1e-2          # Learning rate for Forecast Head
N_SHOTS = 2             # Number of trajectories used for adaptation

# Gating (Safety)
GATE_ALPHA = 20.0       # Steepness of the gate sigmoid [cite: 43]
GATE_TAU = 0.05         # Error threshold (If residual > 0.05, close the gate) [cite: 43]
MC_SAMPLES = 5          # Number of SDE samples for NLL/Uncertainty [cite: 16]

# The Sweep
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_model(sde, head_init, z_init, support, gen, cfg):
    """
    Phase 1: Adaptation (Plasticity).
    Optimizes z and Head to fit the support set.
    """
    start_time = time.time()
    
    # Clone models to avoid overwriting base
    head = copy.deepcopy(head_init)
    head.train()
    
    # Clone Z and enable gradients (Latent Adaptation)
    z_adapted = z_init.clone().detach()
    z_adapted.requires_grad = True
    
    # Optimizer targeting z and head
    optimizer = optim.Adam([
        {'params': head.parameters(), 'lr': LR_HEAD},
        {'params': [z_adapted], 'lr': LR_Z}
    ])
    
    # SDE is frozen (we only adapt parameters z, not weights theta)
    for p in sde.parameters():
        p.requires_grad = False
        
    B, T, D = support.shape
    
    # Simulation Parameters
    T_full = cfg.time_grid.T
    n_steps_full = cfg.time_grid.n_steps
    dt = T_full / n_steps_full
    n_sim = T - 1
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        
        # Expand z for batch
        z_batch = z_adapted.expand(B, -1)
        
        # 1. Simulate dynamics (Physics Match)
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_batch, T_sim, n_sim, x_max, gen)
        
        # 2. Slice to match valid data length
        valid_len = min(traj.shape[1], T)
        traj_sliced = traj[:, :valid_len]
        target_sliced = support[:, :valid_len]
        
        # Loss 1: Path Reconstruction (Physics)
        loss_path = F.mse_loss(traj_sliced, target_sliced)
        
        # Loss 2: Head Prediction (Goal)
        pred_head = head(traj_sliced[:, -1], z_batch)
        loss_head = F.mse_loss(pred_head, target_sliced[:, -1])
        
        # Combined objective
        total_loss = loss_path + loss_head
        
        total_loss.backward()
        optimizer.step()
        
    adapt_time = time.time() - start_time
    head.eval()
    
    return head, z_adapted.detach(), adapt_time

def compute_residual(sde, head, z, support, gen, cfg):
    """
    Phase 2: The Sensor.
    Computes D_res (Eq 7): Residual error of the adapted model on support data.
    """
    B, T, D = support.shape
    T_full = cfg.time_grid.T
    n_steps_full = cfg.time_grid.n_steps
    dt = T_full / n_steps_full
    n_sim = min(T - 1, n_steps_full)
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    
    z_expanded = z.expand(B, -1)
    
    with torch.no_grad():
        # Simulate using the *Adapted* Z
        traj_pred = simulate_neural_sde_batch(sde, support[:, 0], z_expanded, T_sim, n_sim, x_max, gen)
        valid_len = min(traj_pred.shape[1], T)
        
        # Calculate MSE (Residual) 
        residual = F.mse_loss(traj_pred[:, :valid_len], support[:, :valid_len])
        return residual.item()

def gated_inference(encoder, sde, head, support, query, gen, cfg):
    """
    Phase 3: Gated Inference.
    Combines Safe (Mean) and Smart (Adapted) predictions based on Residual.
    """
    # 1. Initial Z (Zero-Shot)
    with torch.no_grad():
        # Use prefix of length up to 50 for initial encoding
        enc_len = min(support.shape[1], 50)
        z_init = encoder(support[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Run Adaptation (Plasticity)
    head_opt, z_opt, adapt_time = adapt_model(sde, head, z_init, support, gen, cfg)
    
    # 3. Compute Residual (The Sensor) on Support Set
    d_res = compute_residual(sde, head_opt, z_opt, support, gen, cfg)
    
    # 4. Compute Gate (The Safety Switch) [cite: 43]
    # g = sigmoid( alpha * (tau - residual) )
    # If residual is LOW (good fit), gate -> 1.
    # If residual is HIGH (bad fit), gate -> 0.
    g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_res))).item()
    
    # 5. Predictions (Mixing)
    B_q = query.shape[0]
    z_smart = z_opt.expand(B_q, -1)
    z_safe = torch.zeros_like(z_smart) # Mean physics (z=0)
    
    T_full = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs
    
    mc_preds = []
    
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            # Path A: Adapted (Smart)
            traj_smart = simulate_neural_sde_batch(sde, query[:, 0], z_smart, T_full, n_steps, x_max, gen)
            
            # Path B: Mean (Safe)
            traj_safe = simulate_neural_sde_batch(sde, query[:, 0], z_safe, T_full, n_steps, x_max, gen)
            
            # Weighted Mixture [cite: 2]
            traj_mix = (1 - g) * traj_safe + g * traj_smart
            mc_preds.append(traj_mix)
            
    # Aggregate MC Samples
    mc_tensor = torch.stack(mc_preds, dim=0)
    pred_mean = mc_tensor.mean(dim=0)
    pred_var = mc_tensor.var(dim=0) + 1e-6
    
    # Metrics
    mse_rollout = F.mse_loss(pred_mean, query).item()
    mse_final = F.mse_loss(pred_mean[:, -1], query[:, -1]).item()
    nll = F.gaussian_nll_loss(pred_mean, query, pred_var).item()
    
    return {
        "gate_value": g,
        "residual_error": d_res,
        "adapt_time": adapt_time,
        "mse_rollout": mse_rollout,
        "mse_final": mse_final,
        "nll": nll
    }

def main():
    device = torch.device(cfg.device)
    print("🛡️ Running Gated Fine-Tuning (Plasticity + Safety)...")
    
    ckpt_path = "checkpoints/meta_epoch_50.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])
    
    encoder.eval(); sde.eval(); head.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    results = []
    
    for regime in ["testA", "testB", "testC"]:
        print(f"\nProcessing {regime}...")
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        
        for theta_id in tqdm(tasks, desc=regime):
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)
            
            for steps in STEPS_SWEEP:
                supp_limited = supp_full[:, :steps, :]
                
                metrics = gated_inference(
                    encoder, sde, head, supp_limited, query, gen, cfg
                )
                
                metrics["regime"] = regime
                metrics["steps_available"] = steps
                results.append(metrics)
                
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/gated_finetuning_full.csv", index=False)
    
    print("\n✅ Results saved to results/gated_finetuning_full.csv")
    print("\n🏆 Final Summary (Test C):")
    print(df[df['regime'] == 'testC'].groupby('steps_available')[
        ['mse_rollout', 'residual_error', 'gate_value']
    ].mean())

if __name__ == "__main__":
    main()