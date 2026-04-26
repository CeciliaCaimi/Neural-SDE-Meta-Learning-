# baselines/adapt_transfer_weak.py
# Weak transfer adaptation: load global SDE+Head (no encoder, z=0)
# and fine-tune HEAD ONLY per task starting from the same z=0,
# with a data-efficiency sweep over available support steps.
import os
import time
import copy
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

# === CONFIG ===
ADAPT_STEPS = 50         # Match Model C / Scratch / GRU for fair comparison
ADAPT_LR = 1e-2          # Head fine-tuning rate
N_SHOTS = 2
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
MC_SAMPLES = 5           # For NLL estimation

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_and_eval_weak_head_only(sde_init, head_init, support, query, config, gen, limit_steps):
    """
    Fine-tunes HEAD ONLY with z=0.
    Returns full metrics: adapt_time, nll, mse_rollout, mse_final.
    """
    device = support.device
    x_dim = config.basis.x_dim
    z_dim = config.latent.latent_dim

    # 1. Start Timer
    start_time = time.time()

    # 2. Clone models (Freeze SDE, Train Head)
    sde = copy.deepcopy(sde_init).to(device)
    head = copy.deepcopy(head_init).to(device)

    sde.eval()
    for p in sde.parameters():
        p.requires_grad = False

    head.train()
    optimizer = optim.Adam(head.parameters(), lr=ADAPT_LR)

    # Simulation setup
    T_full = config.time_grid.T
    n_steps_full = config.time_grid.n_steps
    dt = T_full / n_steps_full
    x_max = config.stability.max_state_abs

    # Slice support to limit_steps
    support_slice = support[:, :limit_steps, :]
    x0_supp = support_slice[:, 0, :]
    target_final = support_slice[:, -1, :]
    z_zeros_supp = torch.zeros(x0_supp.size(0), z_dim, device=device)

    n_sim = limit_steps - 1
    T_train = dt * n_sim

    # 3. Fine-tuning Loop
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        traj = simulate_neural_sde_batch(sde, x0_supp, z_zeros_supp, T_train, n_sim, x_max, gen)
        pred_final = head(traj[:, -1, :], z_zeros_supp)
        loss = F.mse_loss(pred_final, target_final)
        loss.backward()
        optimizer.step()

    adapt_time = time.time() - start_time

    # 4. Evaluation (Monte Carlo for NLL)
    sde.eval()
    head.eval()
    
    x0_q = query[:, 0, :]
    z_zeros_q = torch.zeros(x0_q.size(0), z_dim, device=device)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            # Full simulation 0 -> T
            traj_q = simulate_neural_sde_batch(sde, x0_q, z_zeros_q, T_full, n_steps_full, x_max, gen)
            mc_preds.append(traj_q)

    # Aggregate MC Samples
    mc_tensor = torch.stack(mc_preds, dim=0) # (MC, B, T, D)
    pred_mean = mc_tensor.mean(dim=0)
    pred_var = mc_tensor.var(dim=0) + 1e-6

    # Metrics
    mse_rollout = F.mse_loss(pred_mean, query).item()
    mse_final   = F.mse_loss(pred_mean[:, -1], query[:, -1]).item()
    mse_1step   = F.mse_loss(pred_mean[:, 1],  query[:, 1]).item()
    nll = F.gaussian_nll_loss(pred_mean, query, pred_var).item()

    return {
        "adapt_time":  adapt_time,
        "mse_rollout": mse_rollout,
        "mse_final":   mse_final,
        "mse_1step":   mse_1step,
        "nll":         nll,
    }

def main():
    device = torch.device(cfg.device)
    print("\n" + "=" * 80)
    print("🔬 Weak Transfer Baseline: FULL METRICS (Time, NLL, Rollout)")
    print("=" * 80)

    # Ensure you have run baselines/train_transfer_weak.py first!
    ckpt_path = "checkpoints/transfer_weak_no_encoder.pt"
    if not os.path.exists(ckpt_path):
        print("⚠️  Warning: Checkpoint not found. Using Random Init (for testing only).")
        # In real usage, raise error or ensure file exists
        ckpt_path = None

    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sde.load_state_dict(ckpt["sde"])
        head.load_state_dict(ckpt["head"])

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

        for theta_id in tqdm(ds_supp.metadata["theta_id"].unique(), desc=regime):
            support_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)

            for limit_steps in STEPS_SWEEP:
                metrics = adapt_and_eval_weak_head_only(
                    sde, head, support_full, query, cfg, gen, limit_steps
                )
                
                metrics["regime"] = regime
                metrics["theta_id"] = theta_id
                metrics["steps_available"] = limit_steps
                results.append(metrics)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/transfer_weak_results_full.csv"
    df.to_csv(out_path, index=False)

    print("\n✅ Weak Transfer Full Results:")
    # Print the full table as requested
    pd.set_option('display.max_rows', None)
    print(df.groupby(["regime", "steps_available"])[
        ["mse_rollout", "nll", "adapt_time"]
    ].mean())

if __name__ == "__main__":
    main()