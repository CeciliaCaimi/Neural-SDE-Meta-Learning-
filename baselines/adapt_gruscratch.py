import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from baselines.models_gru import ProbabilisticGRU

# --- CONFIG ---
ADAPT_STEPS = 50
ADAPT_LR = 1e-3       # Standard learning rate
N_SHOTS = 2           # Extreme low data
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201] # The X-axis
ROLLOUT_STEPS = 50    # For Multi-step error

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def recursive_rollout(model, x_seed, n_steps):
    """Autoregressive rollout for Multi-Step Error metric."""
    model.eval()
    trajs = [x_seed] 
    curr_x = x_seed.unsqueeze(1) 
    h = None 
    
    with torch.no_grad():
        for _ in range(n_steps):
            mu, var, h = model(curr_x, h)
            curr_x = mu 
            trajs.append(curr_x.squeeze(1))
    return torch.stack(trajs, dim=1) 

def train_and_eval_scratch(x_dim, support, query, config, limit_steps):
    device = support.device
    
    # 1. FRESH INIT (Random Weights)
    model = ProbabilisticGRU(x_dim).to(device)
    model.train()
    
    # Optimization: Strong weight decay to prevent instant overfitting
    optimizer = optim.Adam(model.parameters(), lr=ADAPT_LR, weight_decay=1e-3)
    
    # Slice Support Data (The Sweep Constraint)
    support_view = support[:, :limit_steps, :]
    inputs = support_view[:, :-1, :]
    targets = support_view[:, 1:, :]
    
    # METRIC: Adaptation Time
    start_time = time.time()
    
    # 2. Train on Support (Overfit to the 2 shots)
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        mu, var, _ = model(inputs)
        loss = F.gaussian_nll_loss(mu, targets, var)
        loss.backward()
        optimizer.step()
        
    adapt_time = time.time() - start_time
    
    # 3. Evaluation
    model.eval()
    with torch.no_grad():
        # A. One-Step Metrics
        q_in = query[:, :-1, :]
        q_target = query[:, 1:, :]
        mu_q, var_q, _ = model(q_in)
        
        nll = F.gaussian_nll_loss(mu_q, q_target, var_q).item()
        mse_one_step = F.mse_loss(mu_q, q_target).item()
        
        # B. Multi-Step Rollout (Simulation Test)
        rollout_start_idx = query.shape[1] - ROLLOUT_STEPS - 1
        x_seed = query[:, rollout_start_idx, :] 
        true_future = query[:, rollout_start_idx : rollout_start_idx + ROLLOUT_STEPS + 1, :]
        
        pred_rollout = recursive_rollout(model, x_seed, ROLLOUT_STEPS)
        mse_rollout = F.mse_loss(pred_rollout, true_future).item()

    return {
        "Adapt_Time": adapt_time,
        "NLL": nll,
        "MSE_OneStep": mse_one_step,
        "MSE_Rollout": mse_rollout
    }

def main():
    device = torch.device(cfg.device)
    print("📉 Evaluating GRU SCRATCH Baseline (Cold Start)")
    print("Hypothesis: This should fail miserably (High Error).")
    
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    out_file = "results/gru_scratch_sweep.csv"
    
    # Clean start
    if os.path.exists(out_file):
        os.remove(out_file)
    
    x_dim = cfg.basis.x_dim

    for regime in ["testA", "testB", "testC"]:
        print(f"Processing {regime}...")
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        task_results = []
        
        for theta_id in tqdm(tasks):
            supp = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)
            
            # --- THE SWEEP ---
            for limit_steps in STEPS_SWEEP:
                metrics = train_and_eval_scratch(x_dim, supp, query, cfg, limit_steps)
                
                metrics["regime"] = regime
                metrics["theta_id"] = theta_id
                metrics["steps_available"] = limit_steps
                task_results.append(metrics)
        
        # Incremental Save
        df_task = pd.DataFrame(task_results)
        mode = 'a' if os.path.exists(out_file) else 'w'
        header = not os.path.exists(out_file)
        df_task.to_csv(out_file, mode=mode, header=header, index=False)

    print(f"\n✅ GRU Scratch Sweep Complete. Saved to {out_file}")
    df = pd.read_csv(out_file)
    print(df.groupby(["regime", "steps_available"])[["MSE_Rollout"]].mean())

if __name__ == "__main__":
    main()