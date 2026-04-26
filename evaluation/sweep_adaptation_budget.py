# evaluation/sweep_adaptation_budget.py
"""
Adaptation-Budget Sensitivity Sweep — Model C
==============================================

Evaluates how rollout MSE changes as the test-time optimisation budget
(number of Adam gradient steps applied to z and the forecast head) is
varied across {10, 25, 50, 100} steps.

The main evaluation (gated_finetuning_regularized.py) fixes ADAPT_STEPS=50.
This sweep verifies that the reported performance advantage of Model C does
not depend on that specific budget choice: a method that only wins at exactly
one budget setting would be a fragile artefact of hyperparameter tuning.

All other hyperparameters (LR, N_SHOTS, BETA_REG, gate formula, MC_SAMPLES)
are held fixed at the values used in the main evaluation.

Outputs
-------
  results/adaptation_budget_metrics.csv   — raw per-task, per-budget results
  plots/adaptation_budget_sensitivity.png — appendix figure

Usage (from project root)
-------------------------
  python -m evaluation.sweep_adaptation_budget

The script supports resuming: existing rows in the CSV are skipped.
"""

import os
import sys
import copy
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

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

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
ADAPT_BUDGETS = [10, 25, 50, 100]   # gradient steps — the sweep axis
STEPS_SWEEP   = [20, 50, 201]       # context lengths to evaluate at each budget

# All other hyperparameters match gated_finetuning_regularized.py exactly.
LR_Z       = 1e-2
LR_HEAD    = 1e-2
N_SHOTS    = 2
BETA_REG   = 0.01
GATE_ALPHA = 20.0
GATE_TAU   = 0.05
MC_SAMPLES = 5

CKPT_PATH   = "checkpoints/meta_epoch_50.pt"
RESULTS_CSV = "results/adaptation_budget_metrics.csv"
OUTPUT_PLOT = "plots/adaptation_budget_sensitivity.png"
SAVE_EVERY  = 20   # flush to disk every N rows

EXPECTED_COLUMNS = [
    "adapt_budget", "steps_available", "regime", "theta_id",
    "gate_value", "residual_error", "adapt_time",
    "mse_rollout", "mse_final", "mse_1step",
    "rmse_rollout", "rmse_final",
    "rmse_per_dim_mean", "rmse_per_dim_max", "nll",
]

REGIME_PALETTE = {"testA": "#2196F3", "testB": "#FF9800", "testC": "#F44336"}
REGIME_LABELS  = {
    "testA": "Regime A (In-dist.)",
    "testB": "Regime B (Mild shift)",
    "testC": "Regime C (Strong shift)",
}
# Line style per context length so all three are visually distinguishable.
STEPS_LINESTYLE = {20: "--", 50: "-", 201: ":"}
STEPS_ALPHA     = {20: 0.45, 50: 0.90, 201: 0.60}


# ---------------------------------------------------------------------------
# Adaptation helpers (parameterised by adapt_steps instead of global constant)
# ---------------------------------------------------------------------------

def _get_task_data(dataset: TrajectoryDataset, theta_id: str,
                   device: torch.device) -> torch.Tensor:
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    data = [dataset[i][0] for i in rows.index.tolist()]
    return torch.stack(data).to(device)


def _adapt(sde: NeuralSDE, head_init: "ForecastHead",
           z_init: torch.Tensor, support: torch.Tensor,
           gen: torch.Generator, adapt_steps: int):
    """
    Run `adapt_steps` Adam steps jointly on (z, head) with the SDE frozen.
    Loss = path MSE + head MSE + L2 regularisation on z.
    Mirrors adapt_model() in gated_finetuning_regularized.py exactly,
    except adapt_steps is explicit rather than the ADAPT_STEPS global.
    """
    head = copy.deepcopy(head_init)
    head.train()
    z = z_init.clone().detach()
    z.requires_grad = True

    opt = optim.Adam(
        [{"params": head.parameters(), "lr": LR_HEAD},
         {"params": [z],               "lr": LR_Z}]
    )
    for p in sde.parameters():
        p.requires_grad = False

    B, T, D = support.shape
    dt    = cfg.time_grid.T / cfg.time_grid.n_steps
    n_sim = T - 1
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs

    t0 = time.time()
    for _ in range(adapt_steps):
        opt.zero_grad()
        z_exp = z.expand(B, -1)
        traj  = simulate_neural_sde_batch(
            sde, support[:, 0], z_exp, T_sim, n_sim, x_max, gen
        )
        vlen      = min(traj.shape[1], T)
        loss_path = F.mse_loss(traj[:, :vlen], support[:, :vlen])
        loss_head = F.mse_loss(
            head(traj[:, vlen - 1], z_exp),
            support[:, vlen - 1],
        )
        loss_reg  = BETA_REG * torch.sum(z ** 2)
        (loss_path + loss_head + loss_reg).backward()
        opt.step()

    return head, z.detach(), time.time() - t0


def _residual(sde: NeuralSDE, head, z: torch.Tensor,
              support: torch.Tensor, gen: torch.Generator) -> float:
    """Support-set MSE after adaptation (same as compute_residual in main script)."""
    B, T, D = support.shape
    dt    = cfg.time_grid.T / cfg.time_grid.n_steps
    n_sim = min(T - 1, cfg.time_grid.n_steps)
    T_sim = dt * n_sim
    z_exp = z.expand(B, -1)
    with torch.no_grad():
        traj = simulate_neural_sde_batch(
            sde, support[:, 0], z_exp, T_sim, n_sim, cfg.stability.max_state_abs, gen
        )
        vlen = min(traj.shape[1], T)
        return F.mse_loss(traj[:, :vlen], support[:, :vlen]).item()


def run_one(encoder, sde, head,
            support_raw: torch.Tensor, query_raw: torch.Tensor,
            gen: torch.Generator, adapt_steps: int,
            target_scaler=None) -> dict:
    """
    Full gated inference for one task at a specific adaptation budget.

    Mirrors gated_inference() in gated_finetuning_regularized.py exactly,
    with adapt_steps as an explicit argument.  All metrics are returned in
    original simulator units (scalers inverted before computing MSE).
    """
    # Normalize if a target scaler is available (two-scalar approach).
    if target_scaler is not None:
        support_in = apply_scaler_to_trajectories(support_raw, target_scaler)
        query_in   = apply_scaler_to_trajectories(query_raw,   target_scaler)
    else:
        support_in = support_raw
        query_in   = query_raw

    # 1. Encode initial z from the support set.
    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        z_init  = encoder(support_in[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Adapt.
    head_opt, z_opt, adapt_time = _adapt(sde, head, z_init, support_in, gen, adapt_steps)

    # 3. Compute gate value using normalised residual (NMSE).
    d_res    = _residual(sde, head_opt, z_opt, support_in, gen)
    data_var = support_in.var().item()
    d_norm   = d_res / (data_var + 1e-8)
    g        = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_norm))).item()

    # 4. MC rollout — gated blend of safe (z=0) and adapted (z_opt) predictions.
    B_q     = query_in.shape[0]
    z_smart = z_opt.expand(B_q, -1)
    z_safe  = torch.zeros_like(z_smart)
    T_full  = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max   = cfg.stability.max_state_abs

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(
                sde, query_in[:, 0], z_smart, T_full, n_steps, x_max, gen
            )
            t_safe = simulate_neural_sde_batch(
                sde, query_in[:, 0], z_safe, T_full, n_steps, x_max, gen
            )
            mc_preds.append((1 - g) * t_safe + g * t_smart)

    mc_tensor = torch.stack(mc_preds)          # (MC, B_q, T, D)
    mean_norm = mc_tensor.mean(dim=0)
    var_norm  = mc_tensor.var(dim=0) + 1e-6

    # 5. Invert to original simulator units before reporting metrics.
    if target_scaler is not None:
        mean       = invert_scaler_to_original(mean_norm, target_scaler)
        query_orig = query_raw
    else:
        mean       = mean_norm
        query_orig = query_in

    mse_rollout = F.mse_loss(mean, query_orig).item()
    mse_final   = F.mse_loss(mean[:, -1], query_orig[:, -1]).item()
    mse_1step   = F.mse_loss(mean[:, 1],  query_orig[:, 1]).item()

    per_dim_mse       = ((mean - query_orig) ** 2).mean(dim=(0, 1))
    per_dim_rmse      = per_dim_mse.sqrt()
    rmse_per_dim_mean = per_dim_rmse.mean().item()
    rmse_per_dim_max  = per_dim_rmse.max().item()

    # NLL in normalised space (where the Gaussian assumption is better calibrated).
    nll = F.gaussian_nll_loss(mean_norm, query_in, var_norm).item()

    return {
        "gate_value":        g,
        "residual_error":    d_res,
        "adapt_time":        adapt_time,
        "mse_rollout":       mse_rollout,
        "mse_final":         mse_final,
        "mse_1step":         mse_1step,
        "rmse_rollout":      mse_rollout ** 0.5,
        "rmse_final":        mse_final   ** 0.5,
        "rmse_per_dim_mean": rmse_per_dim_mean,
        "rmse_per_dim_max":  rmse_per_dim_max,
        "nll":               nll,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots",   exist_ok=True)

    device = torch.device(cfg.device)
    print("=" * 65)
    print("  Adaptation-Budget Sensitivity Sweep — Model C")
    print(f"  Adapt budgets  : {ADAPT_BUDGETS} steps")
    print(f"  Context lengths: {STEPS_SWEEP} steps")
    print(f"  Regimes        : testA, testB, testC")
    print("=" * 65)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Run training.train_meta first."
        )

    ckpt    = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    x_dim   = cfg.basis.x_dim
    z_dim   = cfg.latent.latent_dim

    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde     = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head    = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])
    encoder.eval(); sde.eval(); head.eval()

    source_scaler = ckpt.get("source_scaler", None)
    if source_scaler is not None:
        print("  Source scaler loaded from checkpoint — normalization active.")
    else:
        print("  No source scaler in checkpoint — raw simulator units throughout.")

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    # --- Resume: skip already-completed (budget, steps, regime, task) combos ---
    completed: set[str] = set()
    if os.path.exists(RESULTS_CSV):
        try:
            prev = pd.read_csv(RESULTS_CSV)
            for _, r in prev.iterrows():
                completed.add(
                    f"{int(r['adapt_budget'])}_{int(r['steps_available'])}"
                    f"_{r['regime']}_{r['theta_id']}"
                )
            print(f"  Resuming — {len(completed)} rows already complete.")
        except Exception:
            pass
    else:
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(RESULTS_CSV, index=False)

    buffer: list[dict] = []

    for adapt_budget in ADAPT_BUDGETS:
        print(f"\n── adapt_steps = {adapt_budget} ──────────────────────────")
        for regime in ["testA", "testB", "testC"]:
            try:
                ds_supp  = TrajectoryDataset(index_path, regime, "support")
                ds_query = TrajectoryDataset(index_path, regime, "query")
            except Exception as exc:
                print(f"  WARNING: could not load {regime}: {exc}")
                continue

            tasks = ds_supp.metadata["theta_id"].unique()

            for theta_id in tqdm(tasks, desc=f"budget={adapt_budget} | {regime}"):
                # Load once per task; slice inside the inner loop.
                supp_full = _get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
                query     = _get_task_data(ds_query, theta_id, device)

                # Target scaler fitted on the FULL support set for this task
                # (no query leakage; source_scaler presence gates whether to use it).
                target_scaler = (
                    fit_scaler_on_trajectories(supp_full)
                    if source_scaler is not None else None
                )

                for steps in STEPS_SWEEP:
                    key = f"{adapt_budget}_{steps}_{regime}_{theta_id}"
                    if key in completed:
                        continue

                    metrics = run_one(
                        encoder, sde, head,
                        supp_full[:, :steps], query,
                        gen, adapt_budget, target_scaler,
                    )
                    metrics.update({
                        "adapt_budget":    adapt_budget,
                        "steps_available": steps,
                        "regime":          regime,
                        "theta_id":        theta_id,
                    })
                    buffer.append(metrics)

                if len(buffer) >= SAVE_EVERY:
                    _flush(buffer)
                    buffer = []

    if buffer:
        _flush(buffer)

    print(f"\n✅  Results saved to {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    _print_summary(df)
    _make_plot(df)


def _flush(buffer: list[dict]) -> None:
    pd.DataFrame(buffer, columns=EXPECTED_COLUMNS).to_csv(
        RESULTS_CSV, mode="a", header=False, index=False
    )


def _print_summary(df: pd.DataFrame) -> None:
    print("\nRollout MSE by adapt_budget (steps_available=50, mean across tasks)")
    print("-" * 60)
    sub = df[df["steps_available"] == 50]
    tbl = sub.groupby(["adapt_budget", "regime"])["mse_rollout"].mean().unstack()
    print(tbl.to_string())
    print()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_plot(df: pd.DataFrame) -> None:
    """
    Three-panel figure (one column per regime):
      x-axis : adapt_budget ∈ {10, 25, 50, 100}
      y-axis : mean rollout MSE
      lines  : one per context length in STEPS_SWEEP, with ±95% CI shading
      marker : vertical dotted line at adapt_budget=50 (main-paper default)

    Purpose: show that Model C's performance is stable across a range of
    optimisation budgets, ruling out the possibility that the result is a
    fragile artefact of a single ADAPT_STEPS setting.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        "Adaptation-Budget Sensitivity — Model C\n"
        "Rollout MSE vs. number of test-time gradient steps",
        fontsize=12, fontweight="bold",
    )

    budgets = sorted(df["adapt_budget"].unique())

    for ax, regime in zip(axes, ["testA", "testB", "testC"]):
        sub_r = df[df["regime"] == regime]

        for steps in STEPS_SWEEP:
            sub = sub_r[sub_r["steps_available"] == steps]
            if sub.empty:
                continue

            agg = (
                sub.groupby("adapt_budget")["mse_rollout"]
                .agg(n="count", mean="mean", std="std")
                .reset_index()
            )
            agg["se"]      = agg["std"] / np.sqrt(agg["n"])
            agg["ci_half"] = 1.96 * agg["se"]
            # Clamp lower CI to stay positive on any log-scaled re-use.
            agg["lo"] = np.maximum(
                agg["mean"] - agg["ci_half"],
                agg["mean"] * 1e-3,
            )
            agg["hi"] = agg["mean"] + agg["ci_half"]

            color = REGIME_PALETTE[regime]
            ls    = STEPS_LINESTYLE.get(steps, "-")
            alpha = STEPS_ALPHA.get(steps, 0.7)

            ax.plot(
                agg["adapt_budget"], agg["mean"],
                color=color, linestyle=ls, linewidth=1.8, alpha=alpha,
                marker="o", markersize=5,
                label=f"ctx = {steps} steps",
            )
            ax.fill_between(
                agg["adapt_budget"], agg["lo"], agg["hi"],
                color=color, alpha=0.10,
            )

        # Mark the default ADAPT_STEPS used in the main paper evaluation.
        ax.axvline(
            50, linestyle=":", color="#777", linewidth=1.2,
            label="paper default (50)",
        )

        ax.set_title(REGIME_LABELS[regime], fontsize=10, fontweight="bold")
        ax.set_xlabel("Adaptation Steps (gradient budget)")
        ax.set_ylabel("Rollout MSE")
        ax.set_xticks(ADAPT_BUDGETS)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Figure saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
