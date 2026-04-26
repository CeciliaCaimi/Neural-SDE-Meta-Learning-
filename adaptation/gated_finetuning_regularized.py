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

def gated_inference(encoder, sde, head, support, query, gen, cfg, target_scaler=None):
    """Run gated adaptation and inference for one task.

    Args:
        support: (N_shots, T, D) — raw (unnormalized) support trajectories.
        query:   (N_query, T, D) — raw (unnormalized) query trajectories.
        target_scaler: StandardScaler fitted on the support set for this task.
            When provided, support and query are normalized before the model
            sees them. Predictions are inverted back to original units before
            computing RMSE, so all reported metrics are in simulator units.
            When None, the model operates on raw data throughout.
    """
    # --- 0. Normalize inputs (two-scalar: target scaler fitted on support) ---
    if target_scaler is not None:
        support_in = apply_scaler_to_trajectories(support, target_scaler)
        query_in   = apply_scaler_to_trajectories(query,   target_scaler)
    else:
        support_in = support
        query_in   = query

    # 1. Init
    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        z_init = encoder(support_in[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Adapt (Regularized)
    head_opt, z_opt, adapt_time = adapt_model(sde, head, z_init, support_in, gen, cfg)

    # 3. Gate
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
    # MSE/RMSE are meaningless in a normalized space when the paper reports
    # them as physically interpretable numbers. We invert here so every
    # metric row in the CSV is in the same units as the raw simulator output.
    if target_scaler is not None:
        mean  = invert_scaler_to_original(mean_norm, target_scaler)
        query_orig = query   # already in original units
    else:
        mean  = mean_norm
        query_orig = query_in

    mse_rollout = F.mse_loss(mean, query_orig).item()
    mse_final   = F.mse_loss(mean[:, -1], query_orig[:, -1]).item()

    # Per-dimension RMSE in original units (non-collapsible diagnostic).
    # rmse_per_dim_max exposes dimensional collapse that aggregate MSE hides:
    # a near-constant dimension contributes ≈0 to the average even when all
    # other dimensions have large errors.
    per_dim_mse       = ((mean - query_orig) ** 2).mean(dim=(0, 1))  # (D,)
    per_dim_rmse      = per_dim_mse.sqrt()
    rmse_rollout      = mse_rollout ** 0.5
    rmse_final        = mse_final   ** 0.5
    rmse_per_dim_mean = per_dim_rmse.mean().item()
    rmse_per_dim_max  = per_dim_rmse.max().item()

    # NLL is computed in normalized space (where the Gaussian assumption is
    # more appropriate) using the normalized mean and variance.
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

EXPECTED_COLUMNS = [
    "regime", "theta_id", "steps_available",
    "gate_value", "residual_error", "adapt_time",
    "mse_rollout", "mse_final",
    "rmse_rollout", "rmse_final",
    "rmse_per_dim_mean", "rmse_per_dim_max",
    "nll",
]


def _init_or_repair_csv(path: str) -> set:
    """Return the set of completed keys from path, creating/repairing as needed.

    If the file exists with a stale column schema (e.g. missing the new RMSE
    columns), the missing columns are added as NaN and the file is rewritten
    with the correct schema so future appends don't silently misalign.
    """
    completed_keys = set()
    if not os.path.exists(path):
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
        return completed_keys

    print(f"Resuming from {path}...")
    try:
        existing = pd.read_csv(path)
        # Schema repair: add any missing columns so appended rows align.
        changed = False
        for col in EXPECTED_COLUMNS:
            if col not in existing.columns:
                existing[col] = np.nan
                changed = True
        if changed:
            print(f"  ⚠️  Schema mismatch repaired — added missing columns.")
            existing = existing[EXPECTED_COLUMNS]
            existing.to_csv(path, index=False)

        for _, row in existing.iterrows():
            completed_keys.add(
                f"{row['regime']}_{row['theta_id']}_{int(row['steps_available'])}"
            )
    except Exception as e:
        print(f"  ⚠️  Could not parse existing file ({e}). Starting fresh.")
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)

    return completed_keys


def main():
    device = torch.device(cfg.device)
    print("🛡️  Resumable Gated Finetuning (REGULARIZED + NORMALISED) Started...")

    ckpt = torch.load("checkpoints/meta_epoch_50.pt", map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])
    encoder.eval(); sde.eval(); head.eval()

    # --- Two-scalar approach: load source scaler from checkpoint ---
    # The source scaler was fitted on training data by train_meta.py and
    # saved alongside the model weights. Its presence tells us the model
    # was trained in normalized coordinate space.
    source_scaler = ckpt.get('source_scaler', None)
    if source_scaler is not None:
        print("  ✅ Source scaler loaded from checkpoint. Normalization active.")
    else:
        print("  ⚠️  No source scaler in checkpoint (pre-normalization model).")
        print("     Metrics will be computed in raw simulator units as before.")
        print("     Retrain with train_meta.py to activate full normalization.")

    gen = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    completed_keys = _init_or_repair_csv(RESULTS_PATH)
    buffer = []

    for regime in ["testA", "testB", "testC"]:
        try:
            ds_supp  = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except:
            continue

        tasks = ds_supp.metadata["theta_id"].unique()

        for theta_id in tqdm(tasks, desc=regime):
            needed = any(
                f"{regime}_{theta_id}_{s}" not in completed_keys
                for s in STEPS_SWEEP
            )
            if not needed:
                continue

            # Raw (unnormalized) tensors — normalization is applied inside
            # gated_inference via the per-task target_scaler.
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query     = get_task_data(ds_query, theta_id, device)

            # Fit the TARGET scaler on the support set only (no query leakage).
            # This is the second scalar of the two-scalar approach.
            target_scaler = (
                fit_scaler_on_trajectories(supp_full)
                if source_scaler is not None
                else None
            )

            for steps in STEPS_SWEEP:
                key = f"{regime}_{theta_id}_{steps}"
                if key in completed_keys:
                    continue

                metrics = gated_inference(
                    encoder, sde, head,
                    supp_full[:, :steps], query,
                    gen, cfg,
                    target_scaler=target_scaler,
                )
                metrics.update({
                    "regime": regime,
                    "theta_id": theta_id,
                    "steps_available": steps,
                })
                buffer.append(metrics)

            if len(buffer) >= SAVE_EVERY:
                pd.DataFrame(buffer, columns=EXPECTED_COLUMNS).to_csv(RESULTS_PATH, mode='a', header=False, index=False)
                buffer = []

    if buffer:
        pd.DataFrame(buffer, columns=EXPECTED_COLUMNS).to_csv(RESULTS_PATH, mode='a', header=False, index=False)

    print("\n✅ Regularized Run Complete.")
    full_df = pd.read_csv(RESULTS_PATH)
    print(full_df[full_df['regime'] == 'testC'].groupby('steps_available')[
        ['mse_rollout', 'rmse_rollout', 'rmse_per_dim_mean', 'rmse_per_dim_max',
         'residual_error', 'gate_value']
    ].mean())


if __name__ == "__main__":
    main()