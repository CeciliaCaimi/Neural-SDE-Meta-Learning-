#adaptation/gated_inference 
# adaptation/gated_inference.py
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
K_SUBWINDOWS = 8        
MIN_WINDOW = 20         
MAX_TRAIN_CTX = 50      
GATE_ALPHA = 20.0       
GATE_TAU = 0.1          
N_SHOTS = 2
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201] 
MC_SAMPLES = 5

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def compute_latent_instability(encoder, support_trajs, k=K_SUBWINDOWS):
    """
    Measures instability by bootstrapping prefixes of varying lengths.
    Returns: (variance_scalar, mean_z)
    """
    B, T, D = support_trajs.shape
    effective_max_len = min(T, MAX_TRAIN_CTX)
    
    z_samples_list = []
    with torch.no_grad():
        for _ in range(k):
            # Sample random window length
            if effective_max_len > MIN_WINDOW:
                length = torch.randint(MIN_WINDOW, effective_max_len + 1, (1,)).item()
            else:
                length = min(T, MIN_WINDOW)
            
            batch_view = support_trajs[:, :length, :]
            z = encoder(batch_view)  # (B, z_dim)
            z_samples_list.append(z)
    
    # Stack: (K, B, z_dim)
    z_samples = torch.stack(z_samples_list, dim=0)
    
    # Compute variance per shot: (B, z_dim)
    z_mean = z_samples.mean(dim=0)  # (B, z_dim)
    z_diffs = z_samples - z_mean.unsqueeze(0)  # (K, B, z_dim)
    z_var = (z_diffs ** 2).mean(dim=0).sum(dim=-1)  # (B,) - variance per shot
    
    # Return scalar instability (mean across batch) and mean z
    instability = z_var.mean()  # scalar
    task_z = z_mean.mean(dim=0, keepdim=True)  # (1, z_dim)
    
    return instability, task_z

def compute_support_residual(sde, head, support, z, gen, cfg):
    """
    Computes residual error on the support set after adaptation.
    D̃_res = 1/N sum ||x_{t+1} - x̂_{t+1}(x_t, z)||²
    """
    device = support.device
    B, T, D = support.shape
    
    # Simulate from x_0
    T_full = cfg.time_grid.T
    n_steps_full = cfg.time_grid.n_steps
    dt = T_full / n_steps_full
    n_sim = min(T - 1, n_steps_full)
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    
    z_expanded = z.expand(B, -1)
    
    with torch.no_grad():
        traj_pred = simulate_neural_sde_batch(sde, support[:, 0], z_expanded, T_sim, n_sim, x_max, gen)
        # traj_pred: (B, n_sim+1, D)
        
        # Compare only the observed portion
        valid_len = min(traj_pred.shape[1], T)
        residual = F.mse_loss(traj_pred[:, :valid_len], support[:, :valid_len])
    
    return residual.item()

def adapt_head(head_init, sde, z_star, support, gen, cfg, n_adapt_steps=50, lr=1e-2):
    """
    Fine-tunes the head on the support set. Returns adapted head and wall-clock time.
    """
    start_time = time.time()
    
    head = copy.deepcopy(head_init)
    head.train()
    optimizer = optim.Adam(head.parameters(), lr=lr)
    
    B, T, D = support.shape
    z_expanded = z_star.expand(B, -1)
    
    T_full = cfg.time_grid.T
    n_steps_full = cfg.time_grid.n_steps
    dt = T_full / n_steps_full
    n_sim = T - 1
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    
    for _ in range(n_adapt_steps):
        optimizer.zero_grad()
        
        # Simulate SDE on support horizon
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_expanded, T_sim, n_sim, x_max, gen)
        
        # Align lengths
        valid_len = min(traj.shape[1], T)
        traj_sliced = traj[:, :valid_len]
        target_sliced = support[:, :valid_len]
        
        # Head predicts final step
        pred = head(traj_sliced[:, -1], z_expanded)
        loss = F.mse_loss(pred, target_sliced[:, -1])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    
    adapt_time = time.time() - start_time
    head.eval()
    return head, adapt_time

def compute_onestep_mse(sde, support, z, cfg):
    """
    Computes One-Step MSE using SDE drift: x_{t+1} = x_t + f(x_t, z) * dt
    """
    device = support.device
    B, T, D = support.shape
    
    dt = cfg.time_grid.T / cfg.time_grid.n_steps
    
    # x_t, x_{t+1}
    x_t = support[:, :-1, :].reshape(-1, D)  # (B*(T-1), D)
    x_next = support[:, 1:, :].reshape(-1, D)  # (B*(T-1), D)
    
    # Expand z for each timestep
    z_expanded = z.expand(B, -1).repeat_interleave(T - 1, dim=0)  # (B*(T-1), z_dim)
    
    with torch.no_grad():
        # Get drift
        t_dummy = torch.zeros(x_t.shape[0], 1, device=device)
        drift = sde.f(t_dummy, x_t, z_expanded)  # (B*(T-1), D)
        
        # Euler step
        x_next_pred = x_t + drift * dt
        
        # MSE
        mse = F.mse_loss(x_next_pred, x_next).item()
    
    return mse

def gated_inference(encoder, sde, head, support, query, gen, cfg):
    """
    Complete gated inference pipeline with all metrics.
    """
    device = support.device
    
    # 1. Compute latent instability & z
    instability, task_z = compute_latent_instability(encoder, support)
    
    # 2. Compute support residual (for gate)
    support_residual = compute_support_residual(sde, head, support, task_z, gen, cfg)
    
    # 3. Compute gate: g = σ(α(τ - D̃))
    gate_input = GATE_ALPHA * (GATE_TAU - support_residual)
    g = torch.sigmoid(torch.tensor(gate_input)).item()
    
    # 4. Adapt head (measure time)
    head_adapted, adapt_time = adapt_head(head, sde, task_z, support, gen, cfg)
    
    # 5. Evaluate on query (full trajectory)
    B_q = query.shape[0]
    z_adapted = task_z.expand(B_q, -1)
    z_mean = torch.zeros_like(z_adapted)
    
    T_full = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs
    
    # --- MC Sampling ---
    mc_trajs_adapted = []
    mc_trajs_mean = []
    
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            # Adapted trajectory
            traj_a = simulate_neural_sde_batch(sde, query[:, 0], z_adapted, T_full, n_steps, x_max, gen)
            # Mean trajectory (z=0)
            traj_m = simulate_neural_sde_batch(sde, query[:, 0], z_mean, T_full, n_steps, x_max, gen)
            
            mc_trajs_adapted.append(traj_a)
            mc_trajs_mean.append(traj_m)
    
    # Stack: (MC, B, T, D)
    mc_adapt = torch.stack(mc_trajs_adapted, dim=0)
    mc_mean = torch.stack(mc_trajs_mean, dim=0)
    
    # Gated mixture
    mc_gated = (1 - g) * mc_mean + g * mc_adapt  # (MC, B, T, D)
    
    # Compute statistics
    pred_mean = mc_gated.mean(dim=0)  # (B, T, D)
    pred_var = mc_gated.var(dim=0) + 1e-6  # (B, T, D)
    
    # --- Metrics ---
    mse_rollout = F.mse_loss(pred_mean, query).item()
    mse_final = F.mse_loss(pred_mean[:, -1], query[:, -1]).item()
    nll = F.gaussian_nll_loss(pred_mean, query, pred_var).item()
    
    # One-step check
    mse_onestep = compute_onestep_mse(sde, query, z_adapted, cfg)
    
    return {
        "gate": g,
        "z_instability": instability.item(),
        "support_residual": support_residual,
        "adapt_time": adapt_time,
        "mse_onestep": mse_onestep,
        "mse_rollout": mse_rollout,
        "mse_final": mse_final,
        "nll": nll
    }

def main():
    device = torch.device(cfg.device)
    print("🛡️  Gated Inference with Full Metrics...")
    
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
    
    encoder.eval()
    sde.eval()
    head.eval()
    
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    results = []
    
    for regime in ["testA", "testB", "testC"]:
        print(f"\nProcessing {regime}...")
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support", check_shapes=True)
            ds_query = TrajectoryDataset(index_path, regime, "query", check_shapes=True)
        except Exception as e:
            print(f"Skipping {regime}: {e}")
            continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        
        for theta_id in tqdm(tasks, desc=regime):
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)
            
            for steps in STEPS_SWEEP:
                supp_limited = supp_full[:, :steps, :]
                
                metrics = gated_inference(encoder, sde, head, supp_limited, query, gen, cfg)
                metrics["regime"] = regime
                metrics["steps_available"] = steps
                metrics["theta_id"] = theta_id
                
                results.append(metrics)
    
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/gated_metrics_full.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\n✅ Saved to {out_path}")
    print("\n🏆 Summary (testC):")
    summary = df[df['regime'] == 'testC'].groupby('steps_available')[
        ['mse_rollout', 'mse_final', 'nll', 'adapt_time', 'gate']
    ].mean()
    print(summary)

if __name__ == "__main__":
    main()
