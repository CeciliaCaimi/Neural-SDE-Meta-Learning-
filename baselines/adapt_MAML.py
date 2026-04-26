# baselines/adapt_MAML.py
import os
import time
import copy
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn import functional as F

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# === CONFIG ===
INNER_STEPS = 5     # Keep consistent with training
INNER_LR = 0.01     
N_SHOTS = 2
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
MC_SAMPLES = 5      # Added for NLL

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_maml_per_task(sde_init, head_init, support, query, config, gen, limit_steps):
    """
    Adapts MAML (Weights) and returns FULL metrics (Time, NLL, Rollout).
    """
    device = support.device
    x_dim = config.basis.x_dim
    z_dim = config.latent.latent_dim

    # 1. Start Timer
    start_time = time.time()

    # 2. Clone Init (Fast Weights)
    sde = copy.deepcopy(sde_init)
    head = copy.deepcopy(head_init)
    
    # 3. Inner Loop (Adaptation)
    optimizer = optim.SGD(list(sde.parameters()) + list(head.parameters()), lr=INNER_LR)
    
    # Prepare Support Data
    support_slice = support[:, :limit_steps, :]
    x0_s = support_slice[:, 0, :]
    target_s = support_slice[:, -1, :]
    z_zeros = torch.zeros(x0_s.size(0), z_dim, device=device)
    
    T_full = config.time_grid.T
    n_steps_full = config.time_grid.n_steps
    dt = T_full / n_steps_full
    n_sim = limit_steps - 1
    T_train = dt * n_sim
    x_max = config.stability.max_state_abs

    sde.train()
    head.train()
    
    for _ in range(INNER_STEPS):
        optimizer.zero_grad()
        traj = simulate_neural_sde_batch(sde, x0_s, z_zeros, T_train, n_sim, x_max, gen)
        pred = head(traj[:, -1, :], z_zeros)
        loss = F.mse_loss(pred, target_s)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sde.parameters(), 1.0)
        optimizer.step()
        
    adapt_time = time.time() - start_time
    
    # 4. Evaluation (Monte Carlo for NLL)
    sde.eval()
    head.eval()
    
    T_eval = config.time_grid.T
    n_eval = config.time_grid.n_steps
    
    x0_q = query[:, 0, :]
    z_q = torch.zeros(x0_q.size(0), z_dim, device=device)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            # Full simulation
            traj_q = simulate_neural_sde_batch(sde, x0_q, z_q, T_eval, n_eval, x_max, gen)
            mc_preds.append(traj_q)
            
    # Stack MC samples
    mc_tensor = torch.stack(mc_preds, dim=0)
    pred_mean = mc_tensor.mean(dim=0)
    pred_var = mc_tensor.var(dim=0) + 1e-6
    
    # Metrics
    mse_rollout = F.mse_loss(pred_mean, query).item()
    mse_final = F.mse_loss(pred_mean[:, -1], query[:, -1]).item()
    nll = F.gaussian_nll_loss(pred_mean, query, pred_var).item()
    
    return {
        "adapt_time": adapt_time,
        "mse_rollout": mse_rollout,
        "mse_final": mse_final,
        "nll": nll
    }

def main():
    device = torch.device(cfg.device)
    print("🦎 MAML Baseline: Evaluating with FULL Metrics")
    
    ckpt_path = "checkpoints/maml_nsde_init.pt"
    if not os.path.exists(ckpt_path):
        # Fallback to init if training not done (just to test pipeline)
        print("⚠️ Warning: MAML Checkpoint not found. Using Random Init.")
        # In real run, you MUST have the checkpoint.
        ckpt_path = None 
    
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    sde_init = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head_init = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "sde" in ckpt:
            sde_init.load_state_dict(ckpt["sde"])
            head_init.load_state_dict(ckpt["head"])
        else:
            sde_init.load_state_dict(ckpt)

    gen = torch.Generator(device=device)
    gen.manual_seed(999)
    
    results = []
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    for regime in ["testA", "testB", "testC"]:
        print(f"🎯 Processing {regime}")
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        for theta_id in tqdm(tasks, desc=regime):
            full_support = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            full_query = get_task_data(ds_query, theta_id, device)
            
            for limit_steps in STEPS_SWEEP:
                metrics = adapt_maml_per_task(
                    sde_init, head_init, full_support, full_query, cfg, gen, limit_steps
                )
                
                metrics["regime"] = regime
                metrics["theta_id"] = theta_id
                metrics["steps_available"] = limit_steps
                results.append(metrics)
    
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/maml_results_full.csv", index=False)
    
    print("\n✅ MAML Full Results (Test C):")
    print(df[df['regime']=='testC'].groupby("steps_available")[
        ["mse_rollout", "nll", "adapt_time"]
    ].mean())

if __name__ == "__main__":
    main()