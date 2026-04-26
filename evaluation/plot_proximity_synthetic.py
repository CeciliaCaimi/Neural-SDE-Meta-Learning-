# evaluation/plot_proximity_synthetic.py
#
# Generates plots/manifold_distance_analysis.png — a two-panel figure
# showing how Prediction Error (RMSE) and Gate Value vary with the
# theta-space distance of a test task from the training manifold.
#
# Usage (run from project root):
#   python evaluation/plot_proximity_synthetic.py
#
# Requires:
#   results/gated_regularized_final_fixed.csv   — per-task adaptation metrics
#   results/adaptation_with_distances.csv        — pre-computed theta distances
#     OR  data/meta_params.npz                  — raw SDE parameters (fallback)
#
# Optional (for baseline overlay in the error panel):
#   results/maml_results_full.csv
#   results/transfer_weak_results_full.csv
#
# The distance metric is the L2 norm of the flattened parameter vector
# [theta_b | theta_sigma] to the nearest training task (see proximity_analysis.py).

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from sde_basis.parameters import Theta  # noqa: F401 — needed for unpickling

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
META_PARAMS_PATH = "data/meta_params.npz"
RESULTS_PATH     = "results/gated_regularized_final_fixed.csv"
DISTANCES_PATH   = "results/adaptation_with_distances.csv"
MAML_PATH        = "results/maml_results_full.csv"
TRANSFER_PATH    = "results/transfer_weak_results_full.csv"
OUTPUT_PLOT      = "plots/manifold_distance_analysis.png"

REGIME_PALETTE = {"testA": "#2196F3", "testB": "#FF9800", "testC": "#F44336"}
REGIME_LABELS  = {"testA": "Regime A (In-dist.)",
                  "testB": "Regime B (Mild shift)",
                  "testC": "Regime C (Strong shift)"}
CANONICAL_STEPS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_theta(theta: Theta) -> torch.Tensor:
    return torch.cat([theta.theta_b.flatten(), theta.theta_sigma.flatten()])


def compute_distances(meta_params: dict) -> pd.DataFrame:
    """Compute min-L2 distance from each test theta to the training manifold."""
    train_matrix = torch.stack([flatten_theta(t) for t in meta_params["train"]])
    records = []
    for regime in ["testA", "testB", "testC"]:
        test_thetas = meta_params.get(regime, [])
        if not test_thetas:
            continue
        test_vecs = torch.stack([flatten_theta(t) for t in test_thetas])
        dists     = torch.cdist(test_vecs, train_matrix)
        min_dists = dists.min(dim=1).values.numpy()
        for theta, dist in zip(test_thetas, min_dists):
            records.append({"theta_id": theta.id, "regime": regime,
                            "proximity_score": float(dist)})
    return pd.DataFrame(records)


def _load_baseline(path: str, df_dist: pd.DataFrame, label: str) -> pd.DataFrame | None:
    """Load a baseline CSV, filter to CANONICAL_STEPS, merge distances. Returns None on failure."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df = df[df["steps_available"] == CANONICAL_STEPS].copy()
        df = df.merge(df_dist, on=["theta_id", "regime"], how="inner")
        df = df.dropna(subset=["proximity_score", "mse_rollout"])
        if df.empty:
            print(f"  WARNING: {label} merge produced zero rows — skipping overlay")
            return None
        df["rmse"] = np.sqrt(df["mse_rollout"].clip(lower=0))
        print(f"  Loaded {label} ({len(df)} rows for steps={CANONICAL_STEPS})")
        return df
    except Exception as exc:
        print(f"  WARNING: could not load {label} from {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("plots", exist_ok=True)
    print("Generating manifold distance analysis plot...")

    # --- 1. Distances (cached or computed fresh) ---
    if os.path.exists(DISTANCES_PATH):
        df_dist = pd.read_csv(DISTANCES_PATH)[["theta_id", "regime", "proximity_score"]]
        print(f"  Loaded distances from {DISTANCES_PATH}")
    else:
        if not os.path.exists(META_PARAMS_PATH):
            raise FileNotFoundError(
                f"Missing {DISTANCES_PATH} and {META_PARAMS_PATH}. "
                "Run proximity_analysis.py first."
            )
        meta_params = torch.load(META_PARAMS_PATH, map_location="cpu", weights_only=False)
        print("  Computing distances from meta_params (this takes a moment)...")
        df_dist = compute_distances(meta_params)

    # --- 2. Load Model C results and merge with distances ---
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing: {RESULTS_PATH}")

    df_gated = pd.read_csv(RESULTS_PATH)
    df_model_c = (
        df_gated[df_gated["steps_available"] == CANONICAL_STEPS]
        .merge(df_dist, on=["theta_id", "regime"], how="inner")
        .dropna(subset=["proximity_score", "mse_rollout", "gate_value"])
        .copy()
    )

    if df_model_c.empty:
        raise RuntimeError(
            "Inner join of gated results and distance table produced zero rows.\n"
            f"  gated CSV theta_id sample : {df_gated['theta_id'].unique()[:5].tolist()}\n"
            f"  distances theta_id sample : {df_dist['theta_id'].unique()[:5].tolist()}\n"
            "Check that theta_id formatting matches between the two files, "
            "or delete adaptation_with_distances.csv and re-run proximity_analysis.py."
        )

    df_model_c["rmse"] = np.sqrt(df_model_c["mse_rollout"].clip(lower=0))

    # --- 3. Pearson correlations ---
    r_mse,  p_mse  = stats.pearsonr(df_model_c["proximity_score"], df_model_c["mse_rollout"])
    r_gate, p_gate = stats.pearsonr(df_model_c["proximity_score"], df_model_c["gate_value"])

    print(f"\nPearson r (distance vs MSE):        r = {r_mse:.4f}  (p = {p_mse:.4e})")
    print(f"Pearson r (distance vs gate_value): r = {r_gate:.4f}  (p = {p_gate:.4e})")

    # --- 4. Optional baseline overlays ---
    df_maml     = _load_baseline(MAML_PATH,     df_dist, "MAML")
    df_transfer = _load_baseline(TRANSFER_PATH, df_dist, "Transfer")

    # --- 5. Build two-panel figure ---
    fig, (ax_err, ax_gate_ax) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Continuous Manifold Hypothesis: θ-Space Distance vs. Model Behaviour",
        fontsize=13, fontweight="bold"
    )

    # -- Panel (a): RMSE vs. distance --
    for regime in ["testA", "testB", "testC"]:
        sub = df_model_c[df_model_c["regime"] == regime]
        if sub.empty:
            continue
        ax_err.scatter(sub["proximity_score"], sub["rmse"],
                       c=REGIME_PALETTE[regime], label=REGIME_LABELS[regime],
                       s=55, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
        z  = np.polyfit(sub["proximity_score"], sub["rmse"], 1)
        xr = np.linspace(sub["proximity_score"].min(), sub["proximity_score"].max(), 50)
        ax_err.plot(xr, np.polyval(z, xr), c=REGIME_PALETTE[regime],
                    linestyle="--", linewidth=1.2, alpha=0.7)

    if df_maml is not None:
        ax_err.scatter(df_maml["proximity_score"], df_maml["rmse"],
                       marker="^", c="#9C27B0", s=35, alpha=0.50,
                       edgecolors="none", label="MAML", zorder=2)
    if df_transfer is not None:
        ax_err.scatter(df_transfer["proximity_score"], df_transfer["rmse"],
                       marker="s", c="#607D8B", s=35, alpha=0.50,
                       edgecolors="none", label="Transfer", zorder=2)

    ax_err.set_title(f"(a) RMSE vs. Distance  (steps={CANONICAL_STEPS})", fontsize=11)
    ax_err.set_xlabel("Distance to Training Manifold  (L2, θ-space)")
    ax_err.set_ylabel("RMSE")
    ax_err.annotate(
        f"r = {r_mse:.3f}",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
    )
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.25)

    # -- Panel (b): Gate value vs. distance --
    for regime in ["testA", "testB", "testC"]:
        sub = df_model_c[df_model_c["regime"] == regime]
        if sub.empty:
            continue
        ax_gate_ax.scatter(sub["proximity_score"], sub["gate_value"],
                           c=REGIME_PALETTE[regime], label=REGIME_LABELS[regime],
                           s=55, alpha=0.85, edgecolors="white", linewidths=0.4)
        z  = np.polyfit(sub["proximity_score"], sub["gate_value"], 1)
        xr = np.linspace(sub["proximity_score"].min(), sub["proximity_score"].max(), 50)
        ax_gate_ax.plot(xr, np.polyval(z, xr), c=REGIME_PALETTE[regime],
                        linestyle="--", linewidth=1.2, alpha=0.7)

    ax_gate_ax.axhline(1.0, linestyle=":", color="#555", linewidth=1.0, label="g = 1 (full data)")
    ax_gate_ax.axhline(0.0, linestyle=":", color="#555", linewidth=1.0, label="g = 0 (full prior)")
    ax_gate_ax.set_ylim(-0.05, 1.1)
    ax_gate_ax.set_title(f"(b) Gate Value vs. Distance  (steps={CANONICAL_STEPS})", fontsize=11)
    ax_gate_ax.set_xlabel("Distance to Training Manifold  (L2, θ-space)")
    ax_gate_ax.set_ylabel("Gate Value  g")
    ax_gate_ax.annotate(
        f"r = {r_gate:.3f}",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
    )
    ax_gate_ax.legend(fontsize=8)
    ax_gate_ax.grid(True, alpha=0.25)

    # --- 6. Save ---
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PLOT}")

    # --- 7. Per-regime summary ---
    print(f"\nManifold distance summary (steps={CANONICAL_STEPS})")
    print("-" * 65)
    for regime in ["testA", "testB", "testC"]:
        sub = df_model_c[df_model_c["regime"] == regime]
        if sub.empty:
            continue
        print(f"  {REGIME_LABELS[regime]}")
        print(f"    avg distance = {sub['proximity_score'].mean():.3f} "
              f"± {sub['proximity_score'].std():.3f}")
        print(f"    avg RMSE     = {sub['rmse'].mean():.4f} "
              f"± {sub['rmse'].std():.4f}")
        print(f"    avg gate     = {sub['gate_value'].mean():.4f} "
              f"± {sub['gate_value'].std():.4f}")

    print(f"\nPearson r(distance, RMSE) = {r_mse:.2f}")


if __name__ == "__main__":
    main()
