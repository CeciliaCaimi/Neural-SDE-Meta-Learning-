# evaluation/plot_proximity_synthetic.py
#
# Generates results/manifold_distance_analysis.png — a two-panel figure
# showing how both Prediction Error (RMSE) and Gate Value vary with the
# theta-space distance of a test task from the training manifold.
#
# Usage (run from project root):
#   python evaluation/plot_proximity_synthetic.py
#
# Requires:
#   results/gated_regularized_final_fixed.csv   — per-task adaptation metrics
#   data/meta_params.npz                        — ground-truth SDE parameters
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
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from sde_basis.parameters import Theta  # noqa: F401 — needed for unpickling

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
META_PARAMS_PATH  = "data/meta_params.npz"
RESULTS_PATH      = "results/gated_regularized_final_fixed.csv"
DISTANCES_PATH    = "results/adaptation_with_distances.csv"
OUTPUT_PLOT       = "results/manifold_distance_analysis.png"

REGIME_PALETTE    = {"testA": "#2196F3", "testB": "#FF9800", "testC": "#F44336"}
REGIME_LABELS     = {"testA": "Regime A (In-dist.)", "testB": "Regime B (Mild shift)",
                     "testC": "Regime C (Strong shift)"}
CANONICAL_STEPS   = 50   # adaptation step count used for the comparison panels


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)
    print("Generating manifold distance analysis plot...")

    # --- 1. Load SDE parameters ---
    if not os.path.exists(META_PARAMS_PATH):
        raise FileNotFoundError(f"Missing: {META_PARAMS_PATH}")
    meta_params = torch.load(META_PARAMS_PATH, map_location="cpu", weights_only=False)

    # --- 2. Get per-task distances (use cached file if available) ---
    if os.path.exists(DISTANCES_PATH):
        df_dist = pd.read_csv(DISTANCES_PATH)[["theta_id", "regime", "proximity_score"]]
        print(f"  Loaded distances from {DISTANCES_PATH}")
    else:
        print("  Computing distances from meta_params...")
        df_dist = compute_distances(meta_params)

    # --- 3. Load gated results and filter to canonical step count ---
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing: {RESULTS_PATH}")
    df_gated = pd.read_csv(RESULTS_PATH)
    df_plot  = df_gated[df_gated["steps_available"] == CANONICAL_STEPS].copy()
    df_plot  = df_plot.merge(df_dist, on=["theta_id", "regime"], how="left")
    df_plot  = df_plot.dropna(subset=["proximity_score"])
    df_plot["rmse"] = np.sqrt(df_plot["mse_rollout"].clip(lower=0))

    # --- 4. PCA of full parameter space (for inset / reference) ---
    all_vecs, labels = [], []
    for split, label in [("train", "Train"), ("testA", "Regime A"),
                          ("testB", "Regime B"), ("testC", "Regime C")]:
        if split not in meta_params:
            continue
        for t in meta_params[split]:
            all_vecs.append(flatten_theta(t).numpy())
            labels.append(label)
    X_pca = PCA(n_components=2).fit_transform(np.array(all_vecs))
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Regime"] = labels

    # --- 5. Build figure ---
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        "Continuous Manifold Hypothesis: Distance from Training Manifold vs. Model Behaviour",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_pca  = fig.add_subplot(gs[0])
    ax_rmse = fig.add_subplot(gs[1])
    ax_gate = fig.add_subplot(gs[2])

    # -- Panel 1: PCA of theta-space --
    pca_palette = {"Train": "#BDBDBD", "Regime A": REGIME_PALETTE["testA"],
                   "Regime B": REGIME_PALETTE["testB"], "Regime C": REGIME_PALETTE["testC"]}
    pca_markers = {"Train": "o", "Regime A": "^", "Regime B": "s", "Regime C": "D"}
    for grp, sub in df_pca.groupby("Regime", sort=False):
        ax_pca.scatter(sub["PC1"], sub["PC2"],
                       c=pca_palette.get(grp, "black"),
                       marker=pca_markers.get(grp, "o"),
                       s=35 if grp == "Train" else 55,
                       alpha=0.55 if grp == "Train" else 0.85,
                       label=grp, edgecolors="none")
    ax_pca.set_title("(a) Parameter Space (PCA)", fontsize=11)
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.legend(fontsize=8, markerscale=1.2)
    ax_pca.grid(True, alpha=0.25)

    # -- Panel 2: RMSE vs. distance --
    for regime in ["testA", "testB", "testC"]:
        sub = df_plot[df_plot["regime"] == regime]
        if sub.empty:
            continue
        ax_rmse.scatter(sub["proximity_score"], sub["rmse"],
                        c=REGIME_PALETTE[regime], label=REGIME_LABELS[regime],
                        s=55, alpha=0.8, edgecolors="white", linewidths=0.4)
        # Trend line
        z = np.polyfit(sub["proximity_score"], sub["rmse"], 1)
        xr = np.linspace(sub["proximity_score"].min(), sub["proximity_score"].max(), 50)
        ax_rmse.plot(xr, np.polyval(z, xr), c=REGIME_PALETTE[regime],
                     linestyle="--", linewidth=1.2, alpha=0.7)
    ax_rmse.set_title(f"(b) RMSE vs. Distance  (steps={CANONICAL_STEPS})", fontsize=11)
    ax_rmse.set_xlabel("Distance to Training Manifold  (L2, θ-space)")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, alpha=0.25)

    # -- Panel 3: Gate value vs. distance --
    for regime in ["testA", "testB", "testC"]:
        sub = df_plot[df_plot["regime"] == regime]
        if sub.empty:
            continue
        ax_gate.scatter(sub["proximity_score"], sub["gate_value"],
                        c=REGIME_PALETTE[regime], label=REGIME_LABELS[regime],
                        s=55, alpha=0.8, edgecolors="white", linewidths=0.4)
        z = np.polyfit(sub["proximity_score"], sub["gate_value"], 1)
        xr = np.linspace(sub["proximity_score"].min(), sub["proximity_score"].max(), 50)
        ax_gate.plot(xr, np.polyval(z, xr), c=REGIME_PALETTE[regime],
                     linestyle="--", linewidth=1.2, alpha=0.7)
    ax_gate.axhline(1.0, linestyle=":", color="#555", linewidth=1.0, label="g = 1 (full data)")
    ax_gate.axhline(0.0, linestyle=":", color="#555", linewidth=1.0, label="g = 0 (full prior)")
    ax_gate.set_ylim(-0.05, 1.1)
    ax_gate.set_title(f"(c) Gate Value vs. Distance  (steps={CANONICAL_STEPS})", fontsize=11)
    ax_gate.set_xlabel("Distance to Training Manifold  (L2, θ-space)")
    ax_gate.set_ylabel("Gate Value  g")
    ax_gate.legend(fontsize=8)
    ax_gate.grid(True, alpha=0.25)

    # --- 6. Save ---
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOT}")

    # --- 7. Print summary statistics ---
    print("\nManifold distance summary (steps={})".format(CANONICAL_STEPS))
    print("-" * 65)
    for regime in ["testA", "testB", "testC"]:
        sub = df_plot[df_plot["regime"] == regime]
        if sub.empty:
            continue
        print(f"  {REGIME_LABELS[regime]}")
        print(f"    distance : {sub['proximity_score'].mean():.3f} "
              f"± {sub['proximity_score'].std():.3f}")
        print(f"    RMSE     : {sub['rmse'].mean():.4f} "
              f"± {sub['rmse'].std():.4f}")
        print(f"    gate     : {sub['gate_value'].mean():.4f} "
              f"± {sub['gate_value'].std():.4f}")


if __name__ == "__main__":
    main()
