# adaptation/gated_finetuning_regularized .py
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
ADAPT_STEPS = 50        
LR_Z = 1e-2             
LR_HEAD = 1e-2          
N_SHOTS = 2             

# Safety / Regularization
BETA_REG = 0.01         # Suggestion 3 (Regularization Weight)
GATE_ALPHA = 20.0       
GATE_TAU = 0.05         
MC_SAMPLES = 5          

STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
RESULTS_PATH = "results/gated_regularized_final.csv"
SAVE_EVERY = 5

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_model(sde, head_init, z_init, support, gen, cfg):
    start_time = time.time()
    head = copy.deepcopy(head_init); head.train()
    z_adapted = z_init.clone().detach(); z_adapted.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': head.parameters(), 'lr': LR_HEAD},
        {'params': [z_adapted], 'lr': LR_Z}
    ])
    
    for p in sde.parameters(): p.requires_grad = False
        
    B, T, D = support.shape
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = T - 1; T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        z_batch = z_adapted.expand(B, -1)
        
        # Simulate
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_batch, T_sim, n_sim, x_max, gen)
        valid_len = min(traj.shape[1], T)
        
        # Correct Slicing for Loss
        traj_slice = traj[:, :valid_len, :]        # (B, L, D)
        supp_slice = support[:, :valid_len, :]     # (B, L, D)
        
        # 1. Path Loss (Physics)
        loss_path = F.mse_loss(traj_slice, supp_slice)
        
        # 2. Head Loss (Forecast)
        # We predict using the FINAL state of the simulation slice
        final_state_pred = traj_slice[:, -1, :]    # (B, D)
        final_state_target = supp_slice[:, -1, :]  # (B, D)
        
        head_pred = head(final_state_pred, z_batch)
        loss_head = F.mse_loss(head_pred, final_state_target)
        
        # 3. Regularization (Suggestion 3)
        loss_reg = BETA_REG * torch.sum(z_adapted ** 2)
        
        total_loss = loss_path + loss_head + loss_reg
        
        total_loss.backward()
        optimizer.step()
        
    return head, z_adapted.detach(), time.time() - start_time

def compute_residual(sde, head, z, support, gen, cfg):
    B, T, D = support.shape
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = min(T - 1, n_steps); T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    z_exp = z.expand(B, -1)
    
    with torch.no_grad():
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_exp, T_sim, n_sim, x_max, gen)
        valid_len = min(traj.shape[1], T)
        return F.mse_loss(traj[:, :valid_len], support[:, :valid_len]).item()

def gated_inference(encoder, sde, head, support, query, gen, cfg):
    # 1. Init
    with torch.no_grad():
        enc_len = min(support.shape[1], 50)
        z_init = encoder(support[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Adapt (Regularized)
    head_opt, z_opt, adapt_time = adapt_model(sde, head, z_init, support, gen, cfg)
    
    # 3. Gate
    d_res = compute_residual(sde, head_opt, z_opt, support, gen, cfg)
    g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_res))).item()
    
    # 4. Predict
    B_q = query.shape[0]
    z_smart = z_opt.expand(B_q, -1); z_safe = torch.zeros_like(z_smart)
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(sde, query[:, 0], z_smart, T_full, n_steps, x_max, gen)
            t_safe = simulate_neural_sde_batch(sde, query[:, 0], z_safe, T_full, n_steps, x_max, gen)
            mc_preds.append((1 - g) * t_safe + g * t_smart)
            
    mc_tensor = torch.stack(mc_preds, dim=0)
    mean = mc_tensor.mean(dim=0); var = mc_tensor.var(dim=0) + 1e-6
    
    return {
        "gate_value": g, "residual_error": d_res, "adapt_time": adapt_time,
        "mse_rollout": F.mse_loss(mean, query).item(),
        "mse_final": F.mse_loss(mean[:, -1], query[:, -1]).item(),
        "nll": F.gaussian_nll_loss(mean, query, var).item()
    }

def main():
    device = torch.device(cfg.device)
    print("🛡️  Resumable Gated Finetuning (REGULARIZED + FIXED) Started...")
    
    ckpt = torch.load("checkpoints/meta_epoch_50.pt", map_location=device)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt['encoder']); sde.load_state_dict(ckpt['sde']); head.load_state_dict(ckpt['head'])
    encoder.eval(); sde.eval(); head.eval()
    
    gen = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    completed_keys = set()
    if os.path.exists(RESULTS_PATH):
        print(f"Resuming from {RESULTS_PATH}...")
        try:
            for _, row in pd.read_csv(RESULTS_PATH).iterrows():
                completed_keys.add(f"{row['regime']}_{row['theta_id']}_{int(row['steps_available'])}")
        except: pass
    else:
        pd.DataFrame(columns=["regime", "theta_id", "steps_available", 
                              "gate_value", "residual_error", "adapt_time", 
                              "mse_rollout", "mse_final", "nll"]).to_csv(RESULTS_PATH, index=False)

    buffer = []
    
    for regime in ["testA", "testB", "testC"]:
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        
        for theta_id in tqdm(tasks, desc=regime):
            needed = False
            for steps in STEPS_SWEEP:
                if f"{regime}_{theta_id}_{steps}" not in completed_keys: needed = True; break
            if not needed: continue

            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)
            
            for steps in STEPS_SWEEP:
                key = f"{regime}_{theta_id}_{steps}"
                if key in completed_keys: continue
                
                metrics = gated_inference(encoder, sde, head, supp_full[:, :steps], query, gen, cfg)
                metrics.update({"regime": regime, "theta_id": theta_id, "steps_available": steps})
                buffer.append(metrics)
                
            if len(buffer) >= SAVE_EVERY:
                pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode='a', header=False, index=False)
                buffer = []
                
    if buffer: pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        
    print("\n✅ Regularized Run Complete.")
    full_df = pd.read_csv(RESULTS_PATH)
    print(full_df[full_df['regime']=='testC'].groupby('steps_available')[['mse_rollout', 'residual_error', 'gate_value']].mean())

if __name__ == "__main__":
    main()