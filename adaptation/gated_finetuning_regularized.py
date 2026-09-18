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
from sde_basis.parameterised_sde import drift_true, sigma_diag_true

# === HYPERPARAMETERS ===
ADAPT_STEPS = 50        
LR_Z = 1e-2             
LR_HEAD = 1e-2          
N_SHOTS = 2             

# Safety / Regularization
BETA_REG = 0.01         # Suggestion 3 (Regularization Weight)
GATE_ALPHA = 20.0       
GATE_TAU = 0.02         # Recalibrated for sqrt(N) normalization (was 0.05)
MC_SAMPLES = 5          

STEPS_SWEEP = [20, 40, 50, 80, 100, 120, 201]
RESULTS_PATH = "results/gated_regularized_final.csv"
SAVE_EVERY = 5

def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    idx = rows.index.tolist()
    data = [dataset[i][0] for i in idx]
    return torch.stack(data).to(device)

def adapt_model(sde, head_init, zf_init, zg_init, support, gen, cfg):
    start_time = time.time()
    head = copy.deepcopy(head_init); head.train()
    zf_adapted = zf_init.clone().detach(); zf_adapted.requires_grad = True
    zg_adapted = zg_init.clone().detach(); zg_adapted.requires_grad = True

    optimizer = optim.Adam([
        {'params': head.parameters(), 'lr': LR_HEAD},
        {'params': [zf_adapted], 'lr': LR_Z},
        {'params': [zg_adapted], 'lr': LR_Z}
    ])

    for p in sde.parameters(): p.requires_grad = False

    B, T, D = support.shape
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = T - 1; T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs

    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        zf_batch = zf_adapted.expand(B, -1)
        zg_batch = zg_adapted.expand(B, -1)

        # Simulate
        traj = simulate_neural_sde_batch(sde, support[:, 0], zf_batch, zg_batch, T_sim, n_sim, x_max, gen)
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

        head_pred = head(final_state_pred, zf_batch, zg_batch)
        loss_head = F.mse_loss(head_pred, final_state_target)

        # 3. Regularization (Suggestion 3) — both latents regularised independently
        loss_reg = BETA_REG * (torch.sum(zf_adapted ** 2) + torch.sum(zg_adapted ** 2))

        total_loss = loss_path + loss_head + loss_reg

        total_loss.backward()
        optimizer.step()

    return head, zf_adapted.detach(), zg_adapted.detach(), time.time() - start_time

def compute_residual(sde, head, zf, zg, support, gen, cfg):
    B, T, D = support.shape
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = min(T - 1, n_steps); T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    zf_exp = zf.expand(B, -1)
    zg_exp = zg.expand(B, -1)

    with torch.no_grad():
        traj = simulate_neural_sde_batch(sde, support[:, 0], zf_exp, zg_exp, T_sim, n_sim, x_max, gen)
        valid_len = min(traj.shape[1], T)
        return F.mse_loss(traj[:, :valid_len], support[:, :valid_len]).item()

def compute_mechanism_error(sde, theta, zf, zg, query_in, query_orig, target_scaler):
    """Drift/diffusion identification error (item 1's "drift error"/"diffusion
    error" metrics): compares the adapted model's drift and diffusion
    functions against the true basis-generated drift/diffusion, evaluated
    pointwise at every observed query state (no simulation involved — this
    is a one-step check, the same style as the existing mse_1step metric).

    Args:
        theta: ground-truth Theta for this task (drift/diffusion basis
            coefficients), on the same device as query_in/query_orig.
        zf, zg: the adapted (smart) drift/diffusion latents, shape (1, z_dim).
        query_in: (N_query, T, D) query trajectory, in the same (possibly
            normalized) coordinate frame the SDE nets were trained in.
        query_orig: (N_query, T, D) query trajectory in original simulator
            units — this is the frame drift_true/sigma_diag_true operate in.
        target_scaler: the StandardScaler used to go from query_orig to
            query_in, or None if the model operates on raw data.
    """
    B, T, D = query_orig.shape
    x_raw = query_orig.reshape(-1, D)
    x_in = query_in.reshape(-1, D)
    n = x_in.shape[0]

    zf_exp = zf.expand(n, -1)
    zg_exp = zg.expand(n, -1)
    t_dummy = torch.zeros(n, device=x_in.device)

    with torch.no_grad():
        drift_pred = sde.f(t_dummy, x_in, zf_exp)                       # (n, D)
        diff_pred_mat = sde.g(t_dummy, x_in, zg_exp)                    # (n, D, D)
        diff_pred = torch.diagonal(diff_pred_mat, dim1=-2, dim2=-1)     # (n, D)

    if target_scaler is not None:
        # The model's drift/diffusion were learned in normalized coordinates.
        # For a per-dimension affine normalization x_in = (x_raw - mean) / std,
        # d(x_in)/dt = (1/std) * d(x_raw)/dt, so we rescale back to raw units
        # by multiplying by std elementwise (same for the diffusion amplitude).
        scale = torch.as_tensor(
            target_scaler.scale_, dtype=drift_pred.dtype, device=drift_pred.device
        )
        drift_pred = drift_pred * scale
        diff_pred = diff_pred * scale

    drift_target = drift_true(x_raw, theta)          # (n, D)
    diff_target = sigma_diag_true(x_raw, theta)       # (n, D)

    drift_error = F.mse_loss(drift_pred, drift_target).item()
    diffusion_error = F.mse_loss(diff_pred, diff_target).item()
    return drift_error, diffusion_error

def gated_inference(encoder, sde, head, support, query, gen, cfg, theta, target_scaler=None):
    """Run gated adaptation and inference for one task.

    Args:
        support: (N_shots, T, D) — raw (unnormalized) support trajectories.
        query:   (N_query, T, D) — raw (unnormalized) query trajectories.
        theta: ground-truth Theta (drift/diffusion basis coefficients) for
            this task, used only to compute drift_error/diffusion_error.
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
        zf_all, zg_all = encoder(support_in[:, :enc_len])
        zf_init = zf_all.mean(dim=0, keepdim=True)
        zg_init = zg_all.mean(dim=0, keepdim=True)

    # 2. Adapt (Regularized). z_f and z_g are separate leaf tensors optimized
    # jointly by Adam, so each adapts independently in response to its own
    # gradient (the drift-facing vs. diffusion-facing latent are not tied).
    head_opt, zf_opt, zg_opt, adapt_time = adapt_model(sde, head, zf_init, zg_init, support_in, gen, cfg)

    # 2b. Latent movement under adaptation (item 1's ||Delta z_f||, ||Delta z_g||)
    delta_zf_norm = (zf_opt - zf_init).norm().item()
    delta_zg_norm = (zg_opt - zg_init).norm().item()

    # 3. Gate
    # d_res is in units of (state value)^2 and is scale-dependent. Dividing by
    # the empirical variance of the support data AND sqrt(N) yields a dimensionless
    # NMSE so that GATE_TAU is meaningful regardless of raw vs. normalized data
    # regime or observation length. Without sqrt(N), the gate collapses to 0 at
    # high step counts because d_res grows with sequence length.
    # d_norm ≈ 0: model explains all variance; d_norm = 1: no better than mean.
    # This is still the single scalar gate from the original design — it is
    # applied identically to both z_f and z_g (task 3's learned (a_f, a_g)
    # controller is a separate follow-up, not implemented here).
    d_res = compute_residual(sde, head_opt, zf_opt, zg_opt, support_in, gen, cfg)
    data_var = support_in.var().item()
    N = support_in.shape[1]  # observation length
    d_norm = d_res / (data_var * (N ** 0.5) + 1e-8)  # sqrt(N)-normalized
    g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_norm))).item()

    # 4. Predict (in normalized space if scaler is active)
    B_q = query_in.shape[0]
    zf_smart = zf_opt.expand(B_q, -1); zf_safe = torch.zeros_like(zf_smart)
    zg_smart = zg_opt.expand(B_q, -1); zg_safe = torch.zeros_like(zg_smart)
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(sde, query_in[:, 0], zf_smart, zg_smart, T_full, n_steps, x_max, gen)
            t_safe  = simulate_neural_sde_batch(sde, query_in[:, 0], zf_safe,  zg_safe,  T_full, n_steps, x_max, gen)
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

    # Drift/diffusion identification error (item 1): how well the adapted
    # model's mechanism functions match the true drift/diffusion, evaluated
    # at the query states. Uses the "smart" (adapted) latents — the point is
    # to check what the adapted model *learned*, not the safe fallback.
    drift_error, diffusion_error = compute_mechanism_error(
        sde, theta, zf_opt, zg_opt, query_in, query_orig, target_scaler
    )

    mse_rollout = F.mse_loss(mean, query_orig).item()
    mse_final   = F.mse_loss(mean[:, -1], query_orig[:, -1]).item()
    mse_1step   = F.mse_loss(mean[:, 1],  query_orig[:, 1]).item()

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
        "mse_1step": mse_1step,
        "rmse_rollout": rmse_rollout,
        "rmse_final": rmse_final,
        "rmse_per_dim_mean": rmse_per_dim_mean,
        "rmse_per_dim_max": rmse_per_dim_max,
        "nll": nll,
        "drift_error": drift_error,
        "diffusion_error": diffusion_error,
        "delta_zf_norm": delta_zf_norm,
        "delta_zg_norm": delta_zg_norm,
    }

EXPECTED_COLUMNS = [
    "regime", "theta_id", "steps_available",
    "gate_value", "residual_error", "adapt_time",
    "mse_rollout", "mse_final", "mse_1step",
    "rmse_rollout", "rmse_final",
    "rmse_per_dim_mean", "rmse_per_dim_max",
    "nll",
    "drift_error", "diffusion_error",
    "delta_zf_norm", "delta_zg_norm",
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


def main(regimes=None, checkpoint_path="checkpoints/meta_epoch_50.pt"):
    """
    Args:
        regimes: which "test<Regime>" splits to evaluate (e.g.
            ["testA", "testB", "testC"] or ["testnone", "testdrift_only", ...]).
            Defaults to the legacy combined-shift regimes ["testA", "testB",
            "testC"] if omitted. See the module docstring / --regimes CLI flag.
        checkpoint_path: path to the trained checkpoint (encoder/sde/head
            state dicts + source_scaler), produced by training/train_meta.py.
    """
    if regimes is None:
        regimes = ["testA", "testB", "testC"]

    device = torch.device(cfg.device)
    print("🛡️  Resumable Gated Finetuning (REGULARIZED + NORMALISED) Started...")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
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

    # --- Ground-truth thetas, needed for the drift_error/diffusion_error
    # metrics: meta_params maps "test<Regime>" -> List[Theta], keyed by the
    # same theta_id strings used in index.csv.
    meta_params = torch.load(cfg.paths.meta_params_path, map_location="cpu", weights_only=False)

    gen = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    completed_keys = _init_or_repair_csv(RESULTS_PATH)
    buffer = []

    for regime in regimes:
        try:
            ds_supp  = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except:
            continue

        if regime not in meta_params:
            print(f"  ⚠️  No ground-truth thetas for regime={regime} in "
                  f"{cfg.paths.meta_params_path}; skipping.")
            continue
        theta_by_id = {t.id: t.to(device) for t in meta_params[regime]}

        tasks = ds_supp.metadata["theta_id"].unique()

        for theta_id in tqdm(tasks, desc=regime):
            needed = any(
                f"{regime}_{theta_id}_{s}" not in completed_keys
                for s in STEPS_SWEEP
            )
            if not needed:
                continue

            theta = theta_by_id[theta_id]

            # A theta can end up with zero usable support and/or query
            # trajectories if every simulated rollout was unstable (hit
            # max_state_abs before completing) — more likely at strong OOD
            # shift magnitudes. Skip it rather than crashing on an empty
            # stack; generate_trajectories.py already prints a WARNING for
            # these when they occur.
            n_supp_rows = int((ds_supp.metadata["theta_id"] == theta_id).sum())
            n_query_rows = int((ds_query.metadata["theta_id"] == theta_id).sum())
            if n_supp_rows == 0 or n_query_rows == 0:
                print(f"  ⚠️  Skipping {regime}/{theta_id}: no usable support/query "
                      f"trajectories (support={n_supp_rows}, query={n_query_rows}).")
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
                    gen, cfg, theta,
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
    # Summarize the hardest/last-requested regime (prefer legacy testC when
    # present, since that's the strongest of the original three regimes).
    summary_regime = "testC" if "testC" in regimes else regimes[-1]
    print(f"\nSummary for regime={summary_regime}:")
    print(full_df[full_df['regime'] == summary_regime].groupby('steps_available')[
        ['mse_rollout', 'rmse_rollout', 'rmse_per_dim_mean', 'rmse_per_dim_max',
         'residual_error', 'gate_value',
         'drift_error', 'diffusion_error', 'delta_zf_norm', 'delta_zg_norm']
    ].mean())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run gated test-time adaptation + evaluation for the "
                     "factorised (z_f, z_g) model."
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help=(
            "Comma-separated 'test<Regime>' splits to evaluate, e.g. "
            "'testA,testB,testC' (the default if omitted). Pass 'factorised' "
            "as shorthand for the item-1 regime set generated via "
            "generate_meta_params.py --regimes factorised: " +
            ",".join(f"test{r}" for r in cfg.factorised_test_regimes) +
            ". These splits must already exist in data/index.csv (i.e. "
            "generate_meta_params.py and generate_trajectories.py must have "
            "been run for the same regime names first)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/meta_epoch_50.pt",
        help="Path to the trained checkpoint (default: checkpoints/meta_epoch_50.pt).",
    )
    args = parser.parse_args()

    if args.regimes is None:
        selected_regimes = None
    elif args.regimes.strip() == "factorised":
        selected_regimes = [f"test{r}" for r in cfg.factorised_test_regimes]
    else:
        selected_regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    main(regimes=selected_regimes, checkpoint_path=args.checkpoint)