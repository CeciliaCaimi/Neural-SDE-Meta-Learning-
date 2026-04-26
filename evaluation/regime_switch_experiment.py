# evaluation/regime_switch_experiment.py
"""
Regime-Switch Experiment
========================
Simulates a sudden physics-regime change midway through an online-inference
sequence (testA → testC) and measures how quickly the model's latent
representation adapts.

Experimental protocol
---------------------
For each of N_SEEDS independent (taskA, taskC) task pairs:

  1. Build a "Frankenstein" ground-truth sequence:
        gt = traj_A[:T_PRE]  ||  traj_C[:T_POST + 1]
     This is a *hard shock*: both the hidden state and the underlying physics
     change instantaneously at step T_PRE. It is deliberately adversarial.

  2. Run online sliding-window inference at every step t in
     [CONTEXT_LEN, T_PRE + T_POST):
        context  = gt[t - CONTEXT_LEN : t]           # (CONTEXT_LEN, D)
        z_t      = encoder(context.unsqueeze(0))      # online re-encoding
        x_pred   = gt[t] + sde.f(0, gt[t], z_t) * DT  # Euler step
        mse_t    = MSE(x_pred, gt[t + 1])

  3. Compute per-seed metrics:
        baseline_mse  = mean(mse_t  for t in pre-shock window)
        peak_shock    = max (mse_t  for t in post-shock window)
        recovery_steps = first post-shock index i where mse_t <= baseline_mse
                         (set to T_POST if MSE never returns to baseline)

  4. Aggregate across seeds: report mean ± std for both scalar metrics.

Outputs
-------
  results/regime_switch_metrics.csv   — per-seed raw metrics
  plots/regime_switching_analysis.png — two-panel figure

Run
---
  python -m evaluation.regime_switch_experiment
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from dataloaders.trajectory_datasets import TrajectoryDataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CKPT_PATH    = "checkpoints/meta_epoch_50.pt"
INDEX_PATH   = "data/index.csv"
RESULTS_PATH = "results/regime_switch_metrics.csv"
PLOT_PATH    = "plots/regime_switching_analysis.png"

T_PRE       = 50     # steps of testA ground truth before the shock
T_POST      = 100    # steps of testC ground truth after the shock
CONTEXT_LEN = 20     # sliding-window length fed to the encoder at each step
N_SEEDS     = 5      # number of independent (taskA, taskC) task pairs

# Must match the config that was used during training
DT = cfg.time_grid.T / cfg.time_grid.n_steps   # 1.0 / 200 = 0.005

# Index within the errors array where the shock falls
# errors array spans steps [CONTEXT_LEN, T_PRE + T_POST), so:
SHOCK_IDX = T_PRE - CONTEXT_LEN    # = 30

RESULTS_COLUMNS = [
    "seed", "task_A_id", "task_C_id",
    "baseline_mse", "peak_shock_error", "recovery_steps",
]


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def _load_model(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Run `python -m training.train_meta` first."
        )

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde     = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)

    # Accept both old ('encoder') and new ('encoder_state_dict') checkpoint keys
    enc_key = "encoder_state_dict" if "encoder_state_dict" in ckpt else "encoder"
    sde_key = "sde_state_dict"     if "sde_state_dict"     in ckpt else "sde"
    encoder.load_state_dict(ckpt[enc_key])
    sde.load_state_dict(ckpt[sde_key])

    encoder.eval()
    sde.eval()
    print(f"  Loaded checkpoint: {CKPT_PATH}")
    return encoder, sde


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def _load_regime_trajs(regime: str) -> dict:
    """Return {theta_id: tensor (T+1, D)} — first query trajectory per task."""
    ds = TrajectoryDataset(INDEX_PATH, regime, "query")
    trajs = {}
    for theta_id in ds.metadata["theta_id"].unique():
        rows = ds.metadata[ds.metadata["theta_id"] == theta_id]
        idx  = rows.index.tolist()[0]
        traj, _ = ds[idx]
        trajs[theta_id] = traj
    return trajs


# ---------------------------------------------------------------------------
# SINGLE-SEED EXPERIMENT
# ---------------------------------------------------------------------------
def _run_one_seed(encoder, sde, traj_A, traj_C, device):
    """
    Run the regime-switch experiment for one (taskA, taskC) pair.

    Parameters
    ----------
    traj_A, traj_C : (T+1, D) tensors (on CPU; will be moved to device here)

    Returns
    -------
    errors         : ndarray (T_PRE + T_POST - CONTEXT_LEN,)  per-step MSE
    z_norms        : ndarray same shape  ||z_t||_2 at each step
    baseline_mse   : float   mean MSE over the pre-shock window
    peak_shock     : float   max  MSE over the post-shock window
    recovery_steps : int     first post-shock index where MSE <= baseline_mse,
                             or T_POST if MSE never returns to baseline
    """
    # Concatenate: T_PRE steps of A then T_POST+1 steps of C → (T_PRE+T_POST+1, D)
    # The extra +1 is needed so we can access gt[t+1] at the last step t.
    gt = torch.cat([traj_A[:T_PRE], traj_C[:T_POST + 1]], dim=0).to(device)

    errors  = []
    z_norms = []

    for t in range(CONTEXT_LEN, T_PRE + T_POST):
        context = gt[t - CONTEXT_LEN : t].unsqueeze(0)   # (1, CONTEXT_LEN, D)

        with torch.no_grad():
            z      = encoder(context)                      # (1, z_dim)
            z_norm = torch.norm(z).item()
            z_norms.append(z_norm)

            x_curr = gt[t].unsqueeze(0)                   # (1, D)
            drift  = sde.f(0, x_curr, z)                  # (1, D)
            x_pred = x_curr + drift * DT                  # Euler step
            x_true = gt[t + 1].unsqueeze(0)

            mse = F.mse_loss(x_pred, x_true).item()
            errors.append(mse)

    errors  = np.array(errors,  dtype=np.float32)
    z_norms = np.array(z_norms, dtype=np.float32)

    pre_errors  = errors[:SHOCK_IDX]
    post_errors = errors[SHOCK_IDX:]

    baseline_mse = float(np.mean(pre_errors))  if len(pre_errors)  > 0 else 0.0
    peak_shock   = float(np.max(post_errors))  if len(post_errors) > 0 else 0.0

    # Recovery: first post-shock step where MSE falls back to the pre-shock mean.
    # T_POST is used as a sentinel meaning "did not recover within window".
    recovery_steps = T_POST
    for i, e in enumerate(post_errors):
        if e <= baseline_mse:
            recovery_steps = i
            break

    return errors, z_norms, baseline_mse, peak_shock, recovery_steps


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def _make_plot(
    all_errors, all_z_norms,
    mean_recovery, std_recovery,
    mean_peak, std_peak,
    baseline_mean,
):
    """
    Two-panel figure saved to PLOT_PATH.

    Panel A — Error Trajectory
        Mean MSE ± 1 std across seeds.
        Vertical dashed line at the shock (step T_PRE).
        Green shading over the recovery window (shock → shock + mean_recovery).
        Horizontal dotted line at the pre-shock baseline MSE level.
        Annotation for peak shock error.

    Panel B — Latent Norm Trajectory
        Mean ||z_t|| ± 1 std across seeds.
        Same shock line and recovery shading as Panel A.
    """
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)

    n_err  = all_errors.shape[1]
    # Absolute time-step index for each position in the errors array
    x      = np.arange(CONTEXT_LEN, CONTEXT_LEN + n_err)
    shock_x = T_PRE   # absolute step index where regime changes

    mean_err   = all_errors.mean(axis=0)
    std_err    = all_errors.std(axis=0)
    mean_znorm = all_z_norms.mean(axis=0)
    std_znorm  = all_z_norms.std(axis=0)

    # Absolute step index of the recovery point
    recovery_end_x = min(shock_x + int(round(mean_recovery)),
                         CONTEXT_LEN + n_err - 1)
    recovered = mean_recovery < T_POST

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- Panel A: Error ----
    ax1.fill_between(
        x,
        np.maximum(mean_err - std_err, 0),
        mean_err + std_err,
        alpha=0.20, color="#d62728",
    )
    ax1.plot(x, mean_err, color="#d62728", linewidth=2.0,
             label="Mean one-step MSE (± 1 std)")

    ax1.axvline(shock_x, color="black", linestyle="--", linewidth=1.8,
                label="Regime switch  (testA → testC)")
    ax1.axhline(baseline_mean, color="#1f77b4", linestyle=":",
                linewidth=1.5,
                label=f"Pre-shock baseline  ({baseline_mean:.4f})")

    # Recovery window shading
    rec_label = (
        f"Recovery window  (~{mean_recovery:.1f} ± {std_recovery:.1f} steps)"
        if recovered else
        f"No recovery within {T_POST} steps"
    )
    rec_color = "#2ca02c" if recovered else "#ff7f0e"
    ax1.axvspan(shock_x, recovery_end_x, alpha=0.13, color=rec_color,
                label=rec_label)

    # Peak annotation — placed above the maximum mean-error point post-shock
    post_slice  = mean_err[SHOCK_IDX:]
    peak_rel    = int(np.argmax(post_slice))
    peak_abs_x  = x[SHOCK_IDX + peak_rel]
    peak_val    = post_slice[peak_rel]
    ax1.annotate(
        f"Peak: {mean_peak:.4f} ± {std_peak:.4f}",
        xy=(peak_abs_x, peak_val),
        xytext=(
            min(peak_abs_x + 4, x[-1] - 1),
            peak_val + 0.05 * (mean_err.max() - mean_err.min()),
        ),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    ax1.set_ylabel("One-Step MSE", fontweight="bold")
    ax1.set_title(
        "Regime Switch: Error Trajectory  (testA → testC hard shock)\n"
        f"Peak Shock Error = {mean_peak:.4f} ± {std_peak:.4f}    "
        f"Mean Recovery = {mean_recovery:.1f} ± {std_recovery:.1f} steps",
        fontweight="bold", pad=10,
    )
    ax1.legend(fontsize=9, loc="upper right", frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.25)
    sns.despine(ax=ax1)

    # ---- Panel B: Latent Norm ----
    ax2.fill_between(
        x,
        np.maximum(mean_znorm - std_znorm, 0),
        mean_znorm + std_znorm,
        alpha=0.20, color="#1f77b4",
    )
    ax2.plot(x, mean_znorm, color="#1f77b4", linewidth=2.0,
             label="Mean ||z_t||₂  (± 1 std)")

    ax2.axvline(shock_x, color="black", linestyle="--", linewidth=1.8,
                label="Regime switch")
    ax2.axvspan(shock_x, recovery_end_x, alpha=0.13, color=rec_color)

    ax2.set_xlabel("Time Step", fontweight="bold")
    ax2.set_ylabel("Latent Norm  ||z_t||₂", fontweight="bold")
    ax2.set_title(
        "Latent State Adaptation During Regime Switch",
        fontweight="bold", pad=10,
    )
    ax2.legend(fontsize=9, loc="upper right", frameon=True)
    ax2.grid(True, which="both", ls="--", alpha=0.25)
    sns.despine(ax=ax2)

    plt.tight_layout(h_pad=2.5)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {PLOT_PATH}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Regime-Switch Experiment  (testA → testC)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"  Device: {device}  |  T_PRE={T_PRE}  T_POST={T_POST}  "
        f"CONTEXT_LEN={CONTEXT_LEN}  N_SEEDS={N_SEEDS}"
    )

    encoder, sde = _load_model(device)

    print("  Loading testA trajectories...")
    trajs_A = _load_regime_trajs("testA")
    print("  Loading testC trajectories...")
    trajs_C = _load_regime_trajs("testC")

    ids_A = list(trajs_A.keys())
    ids_C = list(trajs_C.keys())

    n_seeds = min(N_SEEDS, len(ids_A), len(ids_C))
    if n_seeds < N_SEEDS:
        print(
            f"  Warning: only {n_seeds} seeds available "
            f"(testA={len(ids_A)}, testC={len(ids_C)} tasks)."
        )

    # Deterministic task selection
    rng   = np.random.default_rng(cfg.global_seed)
    idx_A = rng.choice(len(ids_A), n_seeds, replace=False)
    idx_C = rng.choice(len(ids_C), n_seeds, replace=False)

    n_err_steps = T_PRE + T_POST - CONTEXT_LEN
    err_list    = []   # each entry: (n_err_steps,) ndarray
    znorm_list  = []
    rows        = []

    for s in tqdm(range(n_seeds), desc="Seeds"):
        task_A_id = ids_A[idx_A[s]]
        task_C_id = ids_C[idx_C[s]]
        traj_A    = trajs_A[task_A_id]
        traj_C    = trajs_C[task_C_id]

        if traj_A.shape[0] < T_PRE or traj_C.shape[0] < T_POST + 1:
            print(
                f"  Seed {s}: trajectory too short "
                f"(A len={traj_A.shape[0]}, C len={traj_C.shape[0]}), skipping."
            )
            continue

        errs, znorms, baseline, peak, rec = _run_one_seed(
            encoder, sde, traj_A, traj_C, device
        )

        err_list.append(errs)
        znorm_list.append(znorms)
        rows.append({
            "seed":             s,
            "task_A_id":        task_A_id,
            "task_C_id":        task_C_id,
            "baseline_mse":     round(float(baseline), 6),
            "peak_shock_error": round(float(peak),     6),
            "recovery_steps":   int(rec),
        })

        print(
            f"  Seed {s}  taskA={task_A_id}  taskC={task_C_id}  |  "
            f"baseline={baseline:.4f}  peak={peak:.4f}  recovery={rec} steps"
        )

    if not rows:
        print("No valid seeds completed. Check data paths and checkpoint.")
        return

    # Save per-seed CSV
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Per-seed metrics saved: {RESULTS_PATH}")

    # Aggregate
    peaks  = df["peak_shock_error"].values
    recs   = df["recovery_steps"].values
    mean_peak     = float(peaks.mean());  std_peak     = float(peaks.std())
    mean_recovery = float(recs.mean());   std_recovery = float(recs.std())
    baseline_mean = float(df["baseline_mse"].mean())

    print()
    print("=" * 65)
    print("  SUMMARY ACROSS SEEDS")
    print("=" * 65)
    print(f"  Peak Shock Error    :  {mean_peak:.4f}  ±  {std_peak:.4f}")
    print(f"  Mean Recovery Time  :  {mean_recovery:.1f}  ±  {std_recovery:.1f}  steps")
    recovered_count = int((recs < T_POST).sum())
    print(f"  Seeds that recovered:  {recovered_count} / {len(rows)}")
    print("=" * 65)

    # Plot
    all_errors  = np.stack(err_list,   axis=0)   # (n_valid_seeds, n_err_steps)
    all_z_norms = np.stack(znorm_list, axis=0)

    _make_plot(
        all_errors, all_z_norms,
        mean_recovery, std_recovery,
        mean_peak, std_peak,
        baseline_mean,
    )


if __name__ == "__main__":
    main()
