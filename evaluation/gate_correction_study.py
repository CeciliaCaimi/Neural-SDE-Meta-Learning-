# evaluation/gate_correction_study.py
"""
Gate Correction Study
=====================
Evaluates Model C under five gate configurations to quantify and visualise
the effect of the residual-normalisation fix (methodology.txt § 6, 13).

Gate variants
-------------
V1  Original    g = σ( α · (τ       − D_res)          )   raw residual, τ fixed
V2  Normalized  g = σ( α · (τ       − D_res/σ²_data)  )   NMSE normalisation
V3  Adaptive τ  g = σ( α · (τ(N)    − D_res/σ²_data)  )   NMSE + horizon-scaled τ

    τ(N) = GATE_TAU · √(N / N_max)

    Rationale: with fewer support steps the adapted model has less data to fit,
    so short-horizon trajectories are inherently noisier. V3 is therefore stricter
    at short horizons (smaller τ → lower gate) and equals V2 at full horizon
    (N = N_max → τ(N) = GATE_TAU).

Reference modes (gate fixed, not a formula)
    always-on   g = 1   fully adapted model, no safety fallback
    always-off  g = 0   safe prior only, adaptation ignored

All five modes share the same adapted parameters (z_opt, head_opt) and the
same MC rollout samples — only the g value fed into the linear mixture
  pred = (1−g)·t_safe + g·t_smart
differs.  This is exact because expectation is linear in g.

Outputs
-------
    results/gate_study_metrics.csv      raw per-task results
    plots/gate_correction_analysis.png  1×3 summary figure
"""

import os
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

CKPT_PATH   = "checkpoints/meta_epoch_50.pt"
RESULTS_CSV = "results/gate_study_metrics.csv"
PLOT_PATH   = "plots/gate_correction_analysis.png"

N_SHOTS     = 2
ADAPT_STEPS = 50
LR_Z        = 1e-2
LR_HEAD     = 1e-2
BETA_REG    = 0.01
MC_SAMPLES  = 5

GATE_ALPHA  = 20.0
GATE_TAU    = 0.05
N_MAX       = 201           # longest horizon in sweep; τ(N_MAX) = GATE_TAU

STEPS_SWEEP = [20, 50, 100, 201]

EXPECTED_COLUMNS = [
    "regime", "theta_id", "steps_available",
    "d_res", "data_var", "d_norm", "tau_v3",
    "gate_v1", "gate_v2", "gate_v3",
    "gate_always_on", "gate_always_off",
    "mse_v1", "mse_v2", "mse_v3",
    "mse_always_on", "mse_always_off",
]

# ---------------------------------------------------------------------------
# Adaptation helpers (self-contained — no import from adaptation/)
# ---------------------------------------------------------------------------

def _adapt(sde, head_init, z_init, support, gen):
    """
    Gradient adaptation of z and head on `support`.
    sde weights are frozen (requires_grad=False) throughout — this is
    intentional and permanent on the shared sde object for this script.
    Returns (head_opt, z_opt): the adapted head copy and latent code.
    """
    head = copy.deepcopy(head_init)
    head.train()
    z = z_init.clone().detach()
    z.requires_grad = True

    opt = optim.Adam(
        [{"params": head.parameters(), "lr": LR_HEAD},
         {"params": [z],               "lr": LR_Z}],
    )
    for p in sde.parameters():
        p.requires_grad = False

    B, T, D = support.shape
    dt      = cfg.time_grid.T / cfg.time_grid.n_steps
    n_sim   = T - 1
    T_sim   = dt * n_sim
    x_max   = cfg.stability.max_state_abs

    for _ in range(ADAPT_STEPS):
        opt.zero_grad()
        z_b  = z.expand(B, -1)
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_b, T_sim, n_sim, x_max, gen)
        vlen = min(traj.shape[1], T)
        loss = (
            F.mse_loss(traj[:, :vlen], support[:, :vlen])
            + F.mse_loss(head(traj[:, vlen - 1], z_b), support[:, vlen - 1])
            + BETA_REG * torch.sum(z ** 2)
        )
        loss.backward()
        opt.step()

    return head, z.detach()


def _residual(sde, head, z, support, gen):
    """Support-set MSE after adaptation (F.mse_loss, averaged over B·T·D)."""
    B, T, D = support.shape
    dt      = cfg.time_grid.T / cfg.time_grid.n_steps
    n_sim   = min(T - 1, cfg.time_grid.n_steps)
    T_sim   = dt * n_sim
    x_max   = cfg.stability.max_state_abs
    z_exp   = z.expand(B, -1)
    with torch.no_grad():
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_exp, T_sim, n_sim, x_max, gen)
        vlen = min(traj.shape[1], T)
        return F.mse_loss(traj[:, :vlen], support[:, :vlen]).item()


# ---------------------------------------------------------------------------
# Gate formulas
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return torch.sigmoid(torch.tensor(x)).item()


def gate_v1(d_res: float, data_var: float, N_steps: int) -> float:
    """Original: raw residual, fixed threshold."""
    return _sigmoid(GATE_ALPHA * (GATE_TAU - d_res))


def gate_v2(d_res: float, data_var: float, N_steps: int) -> float:
    """Normalized: NMSE residual, fixed threshold."""
    return _sigmoid(GATE_ALPHA * (GATE_TAU - d_res / (data_var + 1e-8)))


def gate_v3(d_res: float, data_var: float, N_steps: int) -> float:
    """Adaptive τ: NMSE residual, horizon-scaled threshold.

    τ(N) = GATE_TAU · √(N / N_max).  Smaller at short horizons (stricter),
    equals GATE_TAU at N = N_max.
    """
    tau_n  = GATE_TAU * math.sqrt(N_steps / N_MAX)
    d_norm = d_res / (data_var + 1e-8)
    return _sigmoid(GATE_ALPHA * (tau_n - d_norm))


# ---------------------------------------------------------------------------
# Shared MC rollout
# ---------------------------------------------------------------------------

def _shared_rollout(sde, query, z_opt, gen):
    """
    Pre-compute MC mean trajectories for z_opt (smart) and z=0 (safe).

    Because pred = (1−g)·t_safe + g·t_smart is linear in g, using the
    pre-averaged means is *exactly* equivalent to per-sample mixing for
    the mean prediction.  This avoids running 5 × N_variants simulations
    per task; only 2 × MC_SAMPLES simulations are required.

    Returns (t_smart_mean, t_safe_mean) each (B_q, T, D).
    """
    B_q     = query.shape[0]
    z_smart = z_opt.expand(B_q, -1)
    z_safe  = torch.zeros_like(z_smart)
    T_full  = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max   = cfg.stability.max_state_abs

    smart_buf, safe_buf = [], []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            smart_buf.append(
                simulate_neural_sde_batch(sde, query[:, 0], z_smart, T_full, n_steps, x_max, gen)
            )
            safe_buf.append(
                simulate_neural_sde_batch(sde, query[:, 0], z_safe, T_full, n_steps, x_max, gen)
            )

    t_smart = torch.stack(smart_buf).mean(0)
    t_safe  = torch.stack(safe_buf).mean(0)
    return t_smart, t_safe


def _mse(g: float, t_smart, t_safe, query) -> float:
    pred = (1.0 - g) * t_safe + g * t_smart
    return F.mse_loss(pred, query).item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    data = [dataset[i][0] for i in rows.index.tolist()]
    return torch.stack(data).to(device)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_study(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. Run training first."
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

    gen        = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    records    = []

    for regime in ["testA", "testB", "testC"]:
        try:
            ds_supp  = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except Exception as e:
            print(f"⚠️  Skipping {regime}: {e}")
            continue

        tasks = ds_supp.metadata["theta_id"].unique()

        for theta_id in tqdm(tasks, desc=regime):
            supp_full = _task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query     = _task_data(ds_query, theta_id, device)

            for steps in STEPS_SWEEP:
                support_slice = supp_full[:, :steps]

                # Encode z_init from the available support slice (consistent
                # with gated_inference: enc_len = min(steps, 50))
                with torch.no_grad():
                    enc_len = min(steps, 50)
                    z_init  = encoder(support_slice[:, :enc_len]).mean(0, keepdim=True)

                # Adaptation — shared by all five gate modes
                head_opt, z_opt = _adapt(sde, head, z_init, support_slice, gen)

                # Residual measurements
                d_res    = _residual(sde, head_opt, z_opt, support_slice, gen)
                data_var = support_slice.var().item()
                d_norm   = d_res / (data_var + 1e-8)
                tau_v3   = GATE_TAU * math.sqrt(steps / N_MAX)

                # Gate values
                gates = {
                    "v1":         gate_v1(d_res, data_var, steps),
                    "v2":         gate_v2(d_res, data_var, steps),
                    "v3":         gate_v3(d_res, data_var, steps),
                    "always_on":  1.0,
                    "always_off": 0.0,
                }

                # Shared rollout samples — one call, five gate values
                t_smart, t_safe = _shared_rollout(sde, query, z_opt, gen)

                mses = {k: _mse(g, t_smart, t_safe, query) for k, g in gates.items()}

                records.append({
                    "regime":          regime,
                    "theta_id":        theta_id,
                    "steps_available": steps,
                    "d_res":           d_res,
                    "data_var":        data_var,
                    "d_norm":          d_norm,
                    "tau_v3":          tau_v3,
                    "gate_v1":         gates["v1"],
                    "gate_v2":         gates["v2"],
                    "gate_v3":         gates["v3"],
                    "gate_always_on":  1.0,
                    "gate_always_off": 0.0,
                    "mse_v1":          mses["v1"],
                    "mse_v2":          mses["v2"],
                    "mse_v3":          mses["v3"],
                    "mse_always_on":   mses["always_on"],
                    "mse_always_off":  mses["always_off"],
                })

    df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
    os.makedirs("results", exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ Raw results saved to {RESULTS_CSV}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Visual identity for each mode — (label, hex color, linestyle)
VARIANT_META = {
    "v1":         ("V1 – Original  (raw D_res)",  "#e41a1c", "-"),
    "v2":         ("V2 – Normalized  (NMSE)",      "#377eb8", "--"),
    "v3":         ("V3 – Adaptive τ(N)",            "#4daf4a", "-."),
    "always_on":  ("Always-On   (g = 1)",           "#ff7f00", ":"),
    "always_off": ("Always-Off  (g = 0)",           "#984ea3", ":"),
}


def _ci(series_by_group, n_by_group):
    """Return (mean, lower_95, upper_95) DataFrames."""
    mean = series_by_group.mean()
    std  = series_by_group.std()
    se   = std / np.sqrt(n_by_group)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def make_figure(df):
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.25)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Gate Correction Study — V1 (Original) · V2 (Normalized) · V3 (Adaptive τ)",
        fontweight="bold", y=1.02,
    )

    _panel_a_gate_vs_horizon(axes[0], df)
    _panel_b_gate_vs_residual(axes[1], df)
    _panel_c_mse_comparison(axes[2], df)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"✅ Figure saved to {PLOT_PATH}")
    plt.close(fig)


def _panel_a_gate_vs_horizon(ax, df):
    """
    Panel A — Gate Value vs. Horizon Length
    Mean gate value (± 95% CI across tasks and regimes) for V1, V2, V3
    as a function of steps_available.  Demonstrates that V1 collapses to
    near-zero across all horizons while V2 and V3 remain responsive.
    """
    for var in ("v1", "v2", "v3"):
        label, color, ls = VARIANT_META[var]
        grp  = df.groupby("steps_available")[f"gate_{var}"]
        n    = grp.count()
        mean, lo, hi = _ci(grp, n)

        ax.plot(mean.index, mean.values, color=color, linestyle=ls,
                linewidth=2.2, marker="o", markersize=6, label=label, alpha=0.9)
        ax.fill_between(mean.index, lo, hi, color=color, alpha=0.15)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.55,
               label="g = 0.5 (indifferent)")
    ax.set_xlabel("Support Steps Available", fontweight="bold")
    ax.set_ylabel("Gate Value  g", fontweight="bold")
    ax.set_title("A — Gate Value vs. Horizon\n(mean ± 95% CI across tasks)",
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    sns.despine(ax=ax)


def _panel_b_gate_vs_residual(ax, df):
    """
    Panel B — Gate Value vs. Support Residual D_res
    Scatter of gate values against the raw MSE residual D_res for each task.
    V1 produces a near-constant band at g ≈ 0 regardless of D_res (the
    threshold is too tight for raw data).  V2 and V3 follow a sigmoid-like
    response, opening the gate appropriately when D_res is small relative
    to the data variance.  The theoretical V2 sigmoid (at median data_var)
    is overlaid as a solid curve.
    """
    markers = {"v1": "o", "v2": "s", "v3": "^"}
    plot_df = df.sample(min(len(df), 800), random_state=0) if len(df) > 800 else df

    for var in ("v1", "v2", "v3"):
        _, color, _ = VARIANT_META[var]
        short = var.upper()
        ax.scatter(
            plot_df["d_res"], plot_df[f"gate_{var}"],
            color=color, alpha=0.30, s=16,
            marker=markers[var], edgecolors="none",
            label=f"{short} (scatter)",
        )

    # Theoretical V2 sigmoid at median data_var
    d_grid  = np.linspace(0.0, float(df["d_res"].quantile(0.99)), 250)
    med_var = float(df["data_var"].median())
    g_v2    = torch.sigmoid(
        torch.tensor(GATE_ALPHA * (GATE_TAU - d_grid / (med_var + 1e-8)),
                     dtype=torch.float32)
    ).numpy()
    ax.plot(d_grid, g_v2, color=VARIANT_META["v2"][1],
            linewidth=2.5, linestyle="-", alpha=0.85,
            label=f"V2 sigmoid (median σ²={med_var:.2f})")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Raw Residual  D_res", fontweight="bold")
    ax.set_ylabel("Gate Value  g", fontweight="bold")
    ax.set_title("B — Gate Value vs. Support Residual\n(V1 collapses; V2/V3 stay robust)",
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    sns.despine(ax=ax)


def _panel_c_mse_comparison(ax, df):
    """
    Panel C — Rollout MSE: Gated vs. Ungated across regimes
    Grouped bar chart at the full horizon (steps = N_MAX) showing that
    the corrected gate (V2, V3) achieves lower or equal MSE compared to
    the collapsed gate (V1 ≈ always-off) and that the safety fallback
    (always-off) outperforms blind adaptation (always-on) in OOD regimes,
    validating the gate's utility.
    """
    full     = df[df["steps_available"] == STEPS_SWEEP[-1]].copy()
    modes    = ["v1", "v2", "v3", "always_on", "always_off"]
    regimes  = ["testA", "testB", "testC"]
    n_modes  = len(modes)
    width    = 0.14
    x        = np.arange(len(regimes))

    for i, mode in enumerate(modes):
        label, color, _ = VARIANT_META[mode]
        means = [full[full["regime"] == r][f"mse_{mode}"].mean() for r in regimes]
        stes  = [full[full["regime"] == r][f"mse_{mode}"].sem()  for r in regimes]
        offset = (i - (n_modes - 1) / 2.0) * width
        ax.bar(
            x + offset, means, width,
            yerr=stes, label=label,
            color=color, alpha=0.85,
            capsize=3, error_kw={"linewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_xlabel("Test Regime", fontweight="bold")
    ax.set_ylabel("Rollout MSE  (log scale)", fontweight="bold")
    ax.set_title(
        f"C — Gated vs. Ungated Performance\n(steps = {STEPS_SWEEP[-1]}, mean ± SE)",
        fontweight="bold",
    )
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, loc="upper left")
    sns.despine(ax=ax)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("⚙️   Gate Correction Study")
    print(f"   V1 Original  : g = σ(α·(τ − D_res))")
    print(f"   V2 Normalized: g = σ(α·(τ − D_res/σ²))")
    print(f"   V3 Adaptive  : g = σ(α·(τ(N) − D_res/σ²)),  τ(N) = {GATE_TAU}·√(N/{N_MAX})")
    print(f"   Steps sweep  : {STEPS_SWEEP}")
    print(f"   Device       : {device}\n")

    df = run_study(device)

    print("\n── Mean gate values by variant and horizon ──")
    print(
        df.groupby("steps_available")[["gate_v1", "gate_v2", "gate_v3"]]
        .mean()
        .to_string(float_format="{:.4f}".format)
    )

    print("\n── Mean MSE at full horizon by regime ──")
    full = df[df["steps_available"] == STEPS_SWEEP[-1]]
    print(
        full.groupby("regime")[
            ["mse_v1", "mse_v2", "mse_v3", "mse_always_on", "mse_always_off"]
        ].mean()
        .to_string(float_format="{:.4f}".format)
    )

    make_figure(df)


if __name__ == "__main__":
    main()
