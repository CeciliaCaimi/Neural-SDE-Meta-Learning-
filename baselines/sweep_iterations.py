# baselines/sweep_iterations.py
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# Compare Short Training vs Long Training
ITERATION_SWEEP = [50, 100, 200, 500] 
N_SHOTS = 2

def run_scratch_iterations(x_dim, z_dim, support, query, config, gen, n_iters):
    sde = NeuralSDE(x_dim, z_dim, 64).to(support.device)
    head = ForecastHead(x_dim, z_dim, 64).to(support.device)
    
    # Standard LR
    optimizer = optim.Adam(list(sde.parameters()) + list(head.parameters()), lr=2e-3)
    
    T_full = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs
    z_zeros = torch.zeros(support.size(0), z_dim, device=support.device)
    
    sde.train()
    head.train()
    
    for _ in range(n_iters):
        optimizer.zero_grad()
        traj = simulate_neural_sde_batch(sde, support[:,0,:], z_zeros, T_full, n_steps, x_max, gen)
        pred = head(traj[:, -1, :], z_zeros)
        loss = F.mse_loss(pred, support[:, -1, :]) + F.mse_loss(traj, support)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sde.parameters(), 1.0)
        optimizer.step()
        
    # Evaluate
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
    print("⏳ Running ITERATION SWEEP (Checking Convergence)")
    print(f"Iterations: {ITERATION_SWEEP}")
    
    # Load Test C
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    ds_supp = TrajectoryDataset(index_path, "testC", "support", check_shapes=True)
    ds_query = TrajectoryDataset(index_path, "testC", "query", check_shapes=True)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(999)
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    
    results = []
    
    tasks = ds_supp.metadata["theta_id"].unique()
    
    for theta_id in tqdm(tasks, desc="Test C"):
        full_supp = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
        full_query = get_task_data(ds_query, theta_id, device)
        
        for n_iters in ITERATION_SWEEP:
            mse = run_scratch_iterations(x_dim, z_dim, full_supp, full_query, cfg, gen, n_iters)
            
            results.append({
                "iterations": n_iters,
                "MSE_Scratch": mse
            })
            
    df = pd.DataFrame(results)
    df.to_csv("results/iteration_sweep_results.csv", index=False)
    print("\n✅ Iteration Sweep Results:")
    print(df.groupby("iterations")[["MSE_Scratch"]].mean())

# Helper to get data
def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    data = [dataset[i][0] for i in rows.index]
    return torch.stack(data).to(device)

if __name__ == "__main__":
    main()