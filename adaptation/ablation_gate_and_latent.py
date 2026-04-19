# adaptation/ablation_gate_and_latent.py
"""
Ablation Study: Gate Mechanism and Latent Dimension Sweep
==========================================================

Runs two orthogonal ablations:

  AXIS 1 — Gate mode (uses default z_dim=16 checkpoint):
    'adaptive'    Full safety gate  g = sigmoid(alpha*(tau - D_res))
    'always_on'   Gate forced to 1  — adapted model only, no fallback
    'always_off'  Gate forced to 0  — safe prior z=0 only, no adaptation

  AXIS 2 — Latent dimension (uses adaptive gate mode):
    z_dim ∈ {8, 16, 32}
    Each z_dim requires its own pre-trained checkpoint:
      checkpoints/meta_zdim8_epoch_50.pt   (z_dim = 8)
      checkpoints/meta_epoch_50.pt         (z_dim = 16, default)
      checkpoints/meta_zdim32_epoch_50.pt  (z_dim = 32)

    To generate the non-default checkpoints:
      1. Edit config/base_config.py:  latent_dim = 8  (or 32)
      2. python -m training.train_meta
      3. mv checkpoints/meta_epoch_50.pt checkpoints/meta_zdim8_epoch_50.pt
      4. Restore config/base_config.py to latent_dim = 16

    The script silently skips z_dims whose checkpoints are absent and
    reports which ones are missing at startup.

Outputs
-------
    results/ablation_gate_and_latent.csv   — full per-task metrics
    results/ablation_summary.png           — bar chart summary figure

Usage
-----
    python -m adaptation.ablation_gate_and_latent
"""

import os
import copy
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# ---------------------------------------------------------------------------
# Hyperparameters (match the main evaluation script exactly)
# ---------------------------------------------------------------------------
ADAPT_STEPS  = 50
LR_Z         = 1e-2
LR_HEAD      = 1e-2
N_SHOTS      = 2
BETA_REG     = 0.01
GATE_ALPHA   = 20.0
GATE_TAU     = 0.05
MC_SAMPLES   = 5

# Reduced sweep to keep runtime manageable: start / mid / full-horizon
STEPS_SWEEP  = [20, 50, 201]

# Gate ablation axis
GATE_MODES   = ["adaptive", "always_on", "always_off"]

# Latent dimension ablation axis
LATENT_DIMS_SWEEP = [8, 16, 32]
DEFAULT_Z_DIM     = cfg.latent.latent_dim   # 16

RESULTS_PATH   = "results/ablation_gate_and_latent.csv"
SUMMARY_PLOT   = "results/ablation_summary.png"
SAVE_EVERY     = 10

CKPT_TEMPLATE = {
    DEFAULT_Z_DIM: "checkpoints/meta_epoch_50.pt",
}
for _d in LATENT_DIMS_SWEEP:
    if _d != DEFAULT_Z_DIM:
        CKPT_TEMPLATE[_d] = f"checkpoints/meta_zdim{_d}_epoch_50.pt"


# ---------------------------------------------------------------------------
# Shared adapt / residual helpers (identical to main script)
# ---------------------------------------------------------------------------

def _adapt_model(sde, head_init, z_init, support, gen):
    head = copy.deepcopy(head_init); head.train()
    z_adapted = z_init.clone().detach(); z_adapted.requires_grad = True
    optimizer = optim.Adam([
        {'params': head.parameters(), 'lr': LR_HEAD},
        {'params': [z_adapted], 'lr': LR_Z},
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
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_batch, T_sim, n_sim, x_max, gen)
        vlen = min(traj.shape[1], T)
        loss_path = F.mse_loss(traj[:, :vlen], support[:, :vlen])
        loss_head = F.mse_loss(
            head(traj[:, vlen - 1], z_batch),
            support[:, vlen - 1],
        )
        loss_reg  = BETA_REG * torch.sum(z_adapted ** 2)
        (loss_path + loss_head + loss_reg).backward()
        optimizer.step()

    return head, z_adapted.detach()


def _compute_residual(sde, head, z, support, gen):
    B, T, D = support.shape
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = min(T - 1, n_steps); T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs
    z_exp = z.expand(B, -1)
    with torch.no_grad():
        traj = simulate_neural_sde_batch(sde, support[:, 0], z_exp, T_sim, n_sim, x_max, gen)
        vlen = min(traj.shape[1], T)
        return F.mse_loss(traj[:, :vlen], support[:, :vlen]).item()


# ---------------------------------------------------------------------------
# Core inference function with gate_mode parameter
# ---------------------------------------------------------------------------

def run_inference(encoder, sde, head, support, query, gen, gate_mode: str):
    """
    Args:
        gate_mode:
            'adaptive'   — g = sigmoid(alpha*(tau - D_res))
            'always_on'  — g = 1.0  (never fall back to prior)
            'always_off' — g = 0.0  (always use prior, ignore adaptation)
    Returns:
        dict of per-task scalar metrics
    """
    # 1. Encode initial z
    t0 = time.time()
    with torch.no_grad():
        enc_len = min(support.shape[1], 50)
        z_init = encoder(support[:, :enc_len]).mean(dim=0, keepdim=True)

    # 2. Adapt (even for always_off so adapt_time is comparable)
    head_opt, z_opt = _adapt_model(sde, head, z_init, support, gen)
    adapt_time = time.time() - t0

    # 3. Compute residual (needed for adaptive gate and as diagnostic)
    d_res = _compute_residual(sde, head_opt, z_opt, support, gen)

    # 4. Determine gate value
    if gate_mode == "adaptive":
        g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_res))).item()
    elif gate_mode == "always_on":
        g = 1.0
    elif gate_mode == "always_off":
        g = 0.0
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode!r}")

    # 5. MC rollout
    B_q = query.shape[0]
    z_smart = z_opt.expand(B_q, -1)
    z_safe  = torch.zeros_like(z_smart)
    T_full  = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    x_max   = cfg.stability.max_state_abs

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(sde, query[:, 0], z_smart, T_full, n_steps, x_max, gen)
            t_safe  = simulate_neural_sde_batch(sde, query[:, 0], z_safe,  T_full, n_steps, x_max, gen)
            mc_preds.append((1 - g) * t_safe + g * t_smart)

    mc_tensor = torch.stack(mc_preds, dim=0)
    mean = mc_tensor.mean(dim=0)
    var  = mc_tensor.var(dim=0) + 1e-6

    mse_rollout = F.mse_loss(mean, query).item()
    mse_final   = F.mse_loss(mean[:, -1], query[:, -1]).item()
    per_dim_mse = ((mean - query) ** 2).mean(dim=(0, 1))
    per_dim_rmse = per_dim_mse.sqrt()

    return {
        "gate_value":        g,
        "residual_error":    d_res,
        "adapt_time":        adapt_time,
        "mse_rollout":       mse_rollout,
        "mse_final":         mse_final,
        "rmse_rollout":      mse_rollout ** 0.5,
        "rmse_final":        mse_final   ** 0.5,
        "rmse_per_dim_mean": per_dim_rmse.mean().item(),
        "rmse_per_dim_max":  per_dim_rmse.max().item(),
        "nll":               F.gaussian_nll_loss(mean, query, var).item(),
    }


# ---------------------------------------------------------------------------
# Per-checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(z_dim: int, x_dim: int, device):
    """
    Load encoder / sde / head for a given z_dim.
    Returns (encoder, sde, head) or None if checkpoint missing.
    """
    ckpt_path = CKPT_TEMPLATE[z_dim]
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde     = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head    = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])

    encoder.eval(); sde.eval(); head.eval()
    return encoder, sde, head


def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    data = [dataset[i][0] for i in rows.index.tolist()]
    return torch.stack(data).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "ablation_axis", "z_dim", "gate_mode",
    "regime", "theta_id", "steps_available",
    "gate_value", "residual_error", "adapt_time",
    "mse_rollout", "mse_final",
    "rmse_rollout", "rmse_final",
    "rmse_per_dim_mean", "rmse_per_dim_max",
    "nll",
]


def main():
    device = torch.device(cfg.device)
    print("=" * 70)
    print("  ABLATION STUDY: Gate Mechanism × Latent Dimension")
    print("=" * 70)

    # --- Checkpoint availability ---
    available_zdims = []
    for zdim in LATENT_DIMS_SWEEP:
        path = CKPT_TEMPLATE[zdim]
        if os.path.exists(path):
            available_zdims.append(zdim)
            print(f"  ✅ z_dim={zdim:2d}  checkpoint: {path}")
        else:
            print(f"  ❌ z_dim={zdim:2d}  MISSING: {path}")
            print(f"      → To generate: set latent_dim={zdim} in config/base_config.py,")
            print(f"        run python -m training.train_meta, then rename the checkpoint.")

    if DEFAULT_Z_DIM not in available_zdims:
        print(f"\n❌  Default checkpoint (z_dim={DEFAULT_Z_DIM}) not found. Cannot continue.")
        return

    print()

    # --- Initialise results file ---
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(RESULTS_PATH):
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(RESULTS_PATH, index=False)

    # Load completed keys to support resuming
    completed_keys = set()
    try:
        existing = pd.read_csv(RESULTS_PATH)
        for _, row in existing.iterrows():
            completed_keys.add(
                f"{row['ablation_axis']}_{row['z_dim']}_{row['gate_mode']}"
                f"_{row['regime']}_{row['theta_id']}_{int(row['steps_available'])}"
            )
    except Exception:
        pass

    x_dim = cfg.basis.x_dim
    gen   = torch.Generator(device=device); gen.manual_seed(42)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    buffer = []

    # -----------------------------------------------------------------------
    # AXIS 1: Gate ablation (z_dim=16 fixed, vary gate_mode)
    # -----------------------------------------------------------------------
    print("── AXIS 1: Gate mechanism ablation (z_dim=16) ──")
    enc16, sde16, head16 = load_checkpoint(DEFAULT_Z_DIM, x_dim, device)

    for regime in ["testA", "testB", "testC"]:
        try:
            ds_supp  = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except Exception:
            continue

        tasks = ds_supp.metadata["theta_id"].unique()

        for theta_id in tqdm(tasks, desc=f"Gate ablation | {regime}"):
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query     = get_task_data(ds_query, theta_id, device)

            for gate_mode in GATE_MODES:
                for steps in STEPS_SWEEP:
                    key = (f"gate_{DEFAULT_Z_DIM}_{gate_mode}"
                           f"_{regime}_{theta_id}_{steps}")
                    if key in completed_keys:
                        continue

                    metrics = run_inference(
                        enc16, sde16, head16,
                        supp_full[:, :steps], query,
                        gen, gate_mode,
                    )
                    metrics.update({
                        "ablation_axis":  "gate",
                        "z_dim":          DEFAULT_Z_DIM,
                        "gate_mode":      gate_mode,
                        "regime":         regime,
                        "theta_id":       theta_id,
                        "steps_available": steps,
                    })
                    buffer.append(metrics)

                if len(buffer) >= SAVE_EVERY:
                    pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode="a",
                                                header=False, index=False)
                    buffer = []

    # -----------------------------------------------------------------------
    # AXIS 2: Latent dim sweep (adaptive gate, vary z_dim)
    # -----------------------------------------------------------------------
    print("\n── AXIS 2: Latent dimension sweep (adaptive gate) ──")
    for z_dim in available_zdims:
        if z_dim == DEFAULT_Z_DIM:
            enc, sde_m, head_m = enc16, sde16, head16
        else:
            result = load_checkpoint(z_dim, x_dim, device)
            if result is None:
                continue
            enc, sde_m, head_m = result

        for regime in ["testA", "testB", "testC"]:
            try:
                ds_supp  = TrajectoryDataset(index_path, regime, "support")
                ds_query = TrajectoryDataset(index_path, regime, "query")
            except Exception:
                continue

            tasks = ds_supp.metadata["theta_id"].unique()

            for theta_id in tqdm(tasks, desc=f"z_dim={z_dim} | {regime}"):
                supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
                query     = get_task_data(ds_query, theta_id, device)

                for steps in STEPS_SWEEP:
                    key = f"zdim_{z_dim}_adaptive_{regime}_{theta_id}_{steps}"
                    if key in completed_keys:
                        continue

                    metrics = run_inference(
                        enc, sde_m, head_m,
                        supp_full[:, :steps], query,
                        gen, "adaptive",
                    )
                    metrics.update({
                        "ablation_axis":   "latent_dim",
                        "z_dim":           z_dim,
                        "gate_mode":       "adaptive",
                        "regime":          regime,
                        "theta_id":        theta_id,
                        "steps_available": steps,
                    })
                    buffer.append(metrics)

                if len(buffer) >= SAVE_EVERY:
                    pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode="a",
                                                header=False, index=False)
                    buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(RESULTS_PATH, mode="a", header=False, index=False)

    print(f"\n✅  Results saved to {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Summary table + figure
    # -----------------------------------------------------------------------
    df = pd.read_csv(RESULTS_PATH)

    print("\n── Gate Ablation (steps=201, testC) ──")
    gate_df = df[
        (df["ablation_axis"] == "gate") &
        (df["steps_available"] == 201) &
        (df["regime"] == "testC")
    ]
    print(gate_df.groupby("gate_mode")[["mse_rollout", "rmse_rollout", "gate_value"]].mean())

    zdim_avail = df[df["ablation_axis"] == "latent_dim"]["z_dim"].unique()
    if len(zdim_avail) > 1:
        print("\n── Latent Dim Sweep (steps=201, testC) ──")
        zdim_df = df[
            (df["ablation_axis"] == "latent_dim") &
            (df["steps_available"] == 201) &
            (df["regime"] == "testC")
        ]
        print(zdim_df.groupby("z_dim")[["mse_rollout", "rmse_rollout"]].mean())

    _make_summary_plot(df)


def _make_summary_plot(df):
    """Bar chart: gate ablation (left) and latent dim sweep (right)."""
    import matplotlib
    matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False,
                                "axes.spines.right": False})

    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    # ---- Panel A: Gate ablation per regime ----
    ax_gate = fig.add_subplot(gs[0])
    gate_df = df[
        (df["ablation_axis"] == "gate") &
        (df["steps_available"] == 201)
    ]
    if not gate_df.empty:
        pivot_gate = (
            gate_df.groupby(["regime", "gate_mode"])["rmse_rollout"]
            .mean()
            .unstack("gate_mode")
        )
        cols_order = [c for c in ["adaptive", "always_on", "always_off"]
                      if c in pivot_gate.columns]
        pivot_gate = pivot_gate[cols_order]
        pivot_gate.plot(kind="bar", ax=ax_gate, rot=0, width=0.7,
                        color=["#2c7bb6", "#d7191c", "#abdda4"])
        ax_gate.set_title("Gate Ablation (full horizon, steps=201)", fontweight="bold")
        ax_gate.set_xlabel("Regime")
        ax_gate.set_ylabel("RMSE (original units)")
        ax_gate.legend(title="Gate mode", fontsize=8)

    # ---- Panel B: Latent dim sweep per regime ----
    ax_zdim = fig.add_subplot(gs[1])
    zdim_df = df[
        (df["ablation_axis"] == "latent_dim") &
        (df["steps_available"] == 201)
    ]
    available_zdims = sorted(zdim_df["z_dim"].unique())
    if len(available_zdims) > 0:
        pivot_zdim = (
            zdim_df.groupby(["regime", "z_dim"])["rmse_rollout"]
            .mean()
            .unstack("z_dim")
        )
        pivot_zdim.plot(kind="bar", ax=ax_zdim, rot=0, width=0.7,
                        colormap="viridis")
        ax_zdim.set_title("Latent Dim Sweep (full horizon, steps=201)", fontweight="bold")
        ax_zdim.set_xlabel("Regime")
        ax_zdim.set_ylabel("RMSE (original units)")
        ax_zdim.legend(title="z_dim", fontsize=8)
        if len(available_zdims) < len(LATENT_DIMS_SWEEP):
            missing = set(LATENT_DIMS_SWEEP) - set(available_zdims)
            ax_zdim.text(
                0.5, 0.95,
                f"⚠ Missing checkpoints for z_dim={missing}",
                transform=ax_zdim.transAxes, ha="center", va="top",
                fontsize=8, color="gray",
            )
    else:
        ax_zdim.text(0.5, 0.5, "No latent-dim data available.\nSee README for checkpoint generation.",
                     transform=ax_zdim.transAxes, ha="center", va="center", fontsize=9, color="gray")
        ax_zdim.set_title("Latent Dim Sweep", fontweight="bold")

    plt.suptitle("Ablation Study — Model C", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT, dpi=150, bbox_inches="tight")
    print(f"✅  Summary plot saved to {SUMMARY_PLOT}")
    plt.close()


if __name__ == "__main__":
    main()
