#deepmind/gated_finetuning_regularized_dm.py 
import os
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import (
    TrajectoryDataset,
    fit_scaler_on_trajectories,
    apply_scaler_to_trajectories,
    invert_scaler_to_original,
)
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# === HYPERPARAMETERS ===
# Default Target (Overwritten by benchmark_manager)
TARGET_DATASET = "dm_reacher" 

# Optimization (Plasticity)
ADAPT_STEPS = 50        
LR_Z = 1e-2             
LR_HEAD = 1e-2          
N_SHOTS = 2             # Strict Few-Shot constraint

# Safety / Regularization (Model C)
BETA_REG = 0.01         # Suggestion 3
GATE_ALPHA = 20.0       
GATE_TAU = 0.05         
MC_SAMPLES = 5          

# The Sweep
STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
SAVE_EVERY = 5

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_model(sde, head_init, z_init, support, gen, cfg):
    """
    Phase 1: Regularized Adaptation (Model C).
    """
    start_time = time.time()
    head = copy.deepcopy(head_init); head.train()
    z_adapted = z_init.clone().detach(); z_adapted.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': head.parameters(), 'lr': LR_HEAD},
        {'params': [z_adapted], 'lr': LR_Z}
    ])
    
    # Freeze SDE weights (Meta-Learning assumption: physics laws don't change, only context z)
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
        
        traj_slice = traj[:, :valid_len, :]       
        supp_slice = support[:, :valid_len, :]     
        
        # 1. Physics Loss
        loss_path = F.mse_loss(traj_slice, supp_slice)
        
        # 2. Head Loss
        final_state_pred = traj_slice[:, -1, :]    
        final_state_target = supp_slice[:, -1, :]  
        head_pred = head(final_state_pred, z_batch)
        loss_head = F.mse_loss(head_pred, final_state_target)
        
        # 3. Regularization (Suggestion 3) - The key difference in Model C
        loss_reg = BETA_REG * torch.sum(z_adapted ** 2)
        
        total_loss = loss_path + loss_head + loss_reg
        total_loss.backward()
        optimizer.step()
        
    return head, z_adapted.detach(), time.time() - start_time

def compute_residual(sde, head, z, support, gen, cfg):
    """
    Phase 2: The Sensor. Computes error on support set to drive the Gate.
    """
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

def gated_inference(encoder, sde, head, support, query, gen, cfg, target_scaler=None):
    """Gated inference with optional per-task normalization.

    Args:
        support: (N_shots, T, D) — raw (unnormalized) support trajectories.
        query:   (N_query, T, D) — raw (unnormalized) query trajectories.
        target_scaler: StandardScaler fitted on the support set for this task.
            When provided (two-scalar approach), support and query are normalized
            before the model sees them. Predictions are inverted to original
            simulator units before RMSE is computed, so all CSV values are
            physically interpretable. Critical for DM Control where position
            and velocity dimensions have very different dynamic ranges.
    """
    # --- 0. Normalize inputs ---
    if target_scaler is not None:
        support_in = apply_scaler_to_trajectories(support, target_scaler)
        query_in   = apply_scaler_to_trajectories(query,   target_scaler)
    else:
        support_in = support
        query_in   = query

    # 1. Initial Z
    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        z_init = encoder(support_in[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Adapt (Regularized)
    head_opt, z_opt, adapt_time = adapt_model(sde, head, z_init, support_in, gen, cfg)

    # 3. Compute Gate
    # d_res is in units of (state value)^2 and is scale-dependent. Dividing by
    # the empirical variance of the support data yields a dimensionless NMSE so
    # that GATE_TAU is meaningful regardless of raw vs. normalized data regime.
    # d_norm ≈ 0: model explains all variance; d_norm = 1: no better than mean.
    d_res = compute_residual(sde, head_opt, z_opt, support_in, gen, cfg)
    data_var = support_in.var().item()
    d_norm = d_res / (data_var + 1e-8)
    g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_norm))).item()

    # 4. Predict (in normalized space if scaler is active)
    B_q = query_in.shape[0]
    z_smart = z_opt.expand(B_q, -1); z_safe = torch.zeros_like(z_smart)
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(sde, query_in[:, 0], z_smart, T_full, n_steps, x_max, gen)
            t_safe  = simulate_neural_sde_batch(sde, query_in[:, 0], z_safe,  T_full, n_steps, x_max, gen)
            mc_preds.append((1 - g) * t_safe + g * t_smart)

    mc_tensor = torch.stack(mc_preds, dim=0)
    mean_norm = mc_tensor.mean(dim=0)
    var_norm  = mc_tensor.var(dim=0) + 1e-6

    # --- 5. Invert to original simulator units before computing metrics ---
    # This is especially important for DM Control where position/velocity dims
    # have different scales. Without inversion, MSE is in normalized units and
    # can't be compared across tasks or to physical reference values.
    if target_scaler is not None:
        mean       = invert_scaler_to_original(mean_norm, target_scaler)
        query_orig = query   # already in original units
    else:
        mean       = mean_norm
        query_orig = query_in

    mse_rollout = F.mse_loss(mean, query_orig).item()
    mse_final   = F.mse_loss(mean[:, -1], query_orig[:, -1]).item()

    # Per-dimension RMSE in original units. rmse_per_dim_max exposes collapse
    # on individual observation dimensions that aggregate MSE hides.
    per_dim_mse       = ((mean - query_orig) ** 2).mean(dim=(0, 1))  # (D,)
    per_dim_rmse      = per_dim_mse.sqrt()
    rmse_rollout      = mse_rollout ** 0.5
    rmse_final        = mse_final   ** 0.5
    rmse_per_dim_mean = per_dim_rmse.mean().item()
    rmse_per_dim_max  = per_dim_rmse.max().item()

    nll = F.gaussian_nll_loss(mean_norm, query_in, var_norm).item()

    return {
        "gate_value": g, "residual_error": d_res, "adapt_time": adapt_time,
        "mse_rollout": mse_rollout,
        "mse_final": mse_final,
        "rmse_rollout": rmse_rollout,
        "rmse_final": rmse_final,
        "rmse_per_dim_mean": rmse_per_dim_mean,
        "rmse_per_dim_max": rmse_per_dim_max,
        "nll": nll,
    }

def main():
    device = torch.device(cfg.device)
    print(f"🛡️  DeepMind Adaptation (Regularized Model C)")
    
    # --- 1. CONFIGURATION OVERRIDES ---
    # Allow benchmark_manager to control dataset
    if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'data_root') and "deepmind" in cfg.paths.data_root:
        # data_root is already set by manager, use it
        DATA_ROOT = cfg.paths.data_root
        TASK_NAME = os.path.basename(DATA_ROOT)
    else:
        # Fallback default
        DATA_ROOT = os.path.join("data", "deepmind", TARGET_DATASET)
        cfg.paths.data_root = DATA_ROOT
        TASK_NAME = TARGET_DATASET

    RESULTS_PATH = f"results/dm_{TASK_NAME}_model_c.csv"
    
    # --- 2. CHECKPOINT SELECTION (HONESTY STEP) ---
    if hasattr(cfg, 'ckpt_path_override') and cfg.ckpt_path_override:
        CHECKPOINT_PATH = cfg.ckpt_path_override
    else:
        # Default fallback (assumes you trained manually)
        CHECKPOINT_PATH = f"checkpoints/deepmind/{TASK_NAME}/model_best.pt"

    if not os.path.exists(CHECKPOINT_PATH):
        # Try finding meta_epoch_50.pt
        CHECKPOINT_PATH = f"checkpoints/deepmind/{TASK_NAME}/meta_epoch_50.pt"
        if not os.path.exists(CHECKPOINT_PATH):
             print(f"❌ CRITICAL: No pre-trained model found at {CHECKPOINT_PATH}")
             print("   You must run 'python -m deepmind.train_manager' first!")
             return

    print(f"   📂 Data: {DATA_ROOT}")
    print(f"   💾 Model: {CHECKPOINT_PATH}")
    
    # --- 3. AUTO-DETECT DIMENSION ---
    try:
        sample_path = os.path.join(DATA_ROOT, "train", "task_000_support.pt")
        x_dim = torch.load(sample_path, weights_only=False).shape[-1]
        cfg.basis.x_dim = x_dim
        print(f"   📏 Dimension: {x_dim}")
    except:
        print("⚠️ Could not detect dimension, using config default.")
        x_dim = cfg.basis.x_dim

    # --- 4. INITIALIZE & LOAD ---
    encoder = TrajEncoder(x_dim, cfg.latent.latent_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, cfg.latent.latent_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, cfg.latent.latent_dim, cfg.latent.head_hidden_dim).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])
    encoder.eval(); sde.eval(); head.eval()

    # Two-scalar approach: source scaler signals that the model was trained
    # in normalized coordinate space. Target scaler is fitted per-task below.
    source_scaler = ckpt.get('source_scaler', None)
    if source_scaler is not None:
        print("  ✅ Source scaler loaded. Normalization active.")
    else:
        print("  ⚠️  No source scaler in checkpoint — metrics in raw units (pre-normalization model).")
    
    # --- 5. RUN EVALUATION LOOPS ---
    gen = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(DATA_ROOT, "index.csv")

    # Clean start for results
    if os.path.exists(RESULTS_PATH): os.remove(RESULTS_PATH)
    pd.DataFrame(columns=["regime", "theta_id", "steps_available",
                          "gate_value", "residual_error", "adapt_time",
                          "mse_rollout", "mse_final",
                          "rmse_rollout", "rmse_final",
                          "rmse_per_dim_mean", "rmse_per_dim_max",
                          "nll"]).to_csv(RESULTS_PATH, index=False)

    buffer = []
    
    # Loop over all regimes: Test A (ID), Test B (Extrapolation), Test C (Chaos)
    for regime in ["testA", "testB", "testC"]: 
        try:
            ds_supp = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except: 
            print(f"⚠️ Skipping {regime} (not found in index).")
            continue
        
        tasks = ds_supp.metadata["theta_id"].unique()
        
        for theta_id in tqdm(tasks, desc=f"{TASK_NAME} [{regime}]"):
            # Strict Few-Shot: Only see first N_SHOTS
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query     = get_task_data(ds_query, theta_id, device)

            # Fit the TARGET scaler on the support set only (prevents query leakage).
            # For DM Control this is especially important: velocity and position
            # dimensions can differ by 10-100x in magnitude.
            target_scaler = (
                fit_scaler_on_trajectories(supp_full)
                if source_scaler is not None
                else None
            )

            for steps in STEPS_SWEEP:
                # Cap steps if data is shorter than sweep requirement
                actual_steps = min(steps, supp_full.shape[1])

                metrics = gated_inference(
                    encoder, sde, head,
                    supp_full[:, :actual_steps], query,
                    gen, cfg,
                    target_scaler=target_scaler,
                )
                metrics.update({"regime": regime, "theta_id": theta_id, "steps_available": steps})
                buffer.append(metrics)
                
            # Incremental Save
            if len(buffer) >= SAVE_EVERY:
                pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode='a', header=False, index=False)
                buffer = []
                
    if buffer:
        pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        
    print(f"\n✅ Results saved to {RESULTS_PATH}")
    full_df = pd.read_csv(RESULTS_PATH)
    # Quick Summary Print
    print(full_df.groupby(['regime', 'steps_available'])[
        ['mse_rollout', 'rmse_rollout', 'rmse_per_dim_mean', 'rmse_per_dim_max', 'gate_value']
    ].mean())

if __name__ == "__main__":
    main()