# evaluation/plot_gate_calibration.py
"""
Gate Calibration Plot
=====================

Plots the empirical gate value g against the residual error D_res, overlaid
with the theoretical sigmoid curve:

    g = sigmoid(alpha * (tau - D_res))    [alpha=20, tau=0.02]

A monotonically decreasing relationship — higher residual error leads to a
lower gate value — validates that the safety mechanism correctly de-weights
the adapted model when adaptation quality is poor. This is Figure X
("Gate Calibration") in the paper.

The plot also shows the distribution of gate values per regime (testA/B/C)
as a marginal histogram, illustrating that out-of-distribution regimes
(testC) trigger lower gate values than in-distribution regimes (testA).

Inputs
------
    results/gated_regularized_final_fixed.csv   — main evaluation results

    Falls back to results/gated_regularized_final.csv if the fixed file is
    absent, but prints a warning since that file has known schema issues.

Outputs
-------
    results/gate_calibration.png              — main scatter + sigmoid overlay
    results/gate_calibration_by_steps.png     — faceted by steps_available

Usage
-----
    python -m evaluation.plot_gate_calibration
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants (must match adaptation/gated_finetuning_regularized.py exactly)
# ---------------------------------------------------------------------------
GATE_ALPHA = 20.0
GATE_TAU   = 0.02

RESULTS_CANDIDATES = [
    "results/gated_regularized_final_fixed.csv",
    "results/gated_regularized_final.csv",
]
OUT_MAIN   = "results/gate_calibration.png"
OUT_FACET  = "results/gate_calibration_by_steps.png"

REGIME_COLORS  = {"testA": "#2c7bb6", "testB": "#fdae61", "testC": "#d7191c"}
REGIME_LABELS  = {"testA": "Test A (in-dist.)", "testB": "Test B (extrap.)", "testC": "Test C (OOD)"}
REGIME_ORDER   = ["testA", "testB", "testC"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    for path in RESULTS_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Validate required columns
            required = {"gate_value", "residual_error", "regime"}
            if not required.issubset(df.columns):
                print(f"  ⚠️  {path} is missing columns {required - set(df.columns)}. Trying next.")
                continue

            # Coerce to numeric (guard against schema-shifted files)
            for col in ["gate_value", "residual_error"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["gate_value", "residual_error", "regime"])

            # Keep only rows with sensible gate values (0 ≤ g ≤ 1)
            n_before = len(df)
            df = df[(df["gate_value"] >= 0) & (df["gate_value"] <= 1)]
            if len(df) < n_before:
                print(f"  ⚠️  Dropped {n_before - len(df)} rows with gate_value outside [0, 1].")

            if df.empty:
                print(f"  ❌  {path} contains no valid rows after cleaning.")
                continue

            print(f"  ✅  Loaded {len(df)} rows from {path}")
            return df

    print("❌  No valid results file found. Run adaptation.gated_finetuning_regularized first.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Theoretical sigmoid
# ---------------------------------------------------------------------------

def theoretical_gate(d_res: np.ndarray) -> np.ndarray:
    """g = sigmoid(alpha * (tau - d_res))"""
    return 1.0 / (1.0 + np.exp(-GATE_ALPHA * (GATE_TAU - d_res)))


# ---------------------------------------------------------------------------
# Main scatter + sigmoid overlay plot
# ---------------------------------------------------------------------------

def plot_main(df: pd.DataFrame):
    """
    Two-panel figure:
      Left  — scatter g vs D_res, colored by regime, theoretical curve overlaid
      Right — violin / strip of gate values by regime
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    sns.set_style("whitegrid")

    ax_scatter = axes[0]
    ax_violin  = axes[1]

    # ---- Left: scatter + sigmoid ----
    d_max = df["residual_error"].quantile(0.99) * 1.05   # trim extreme outliers for display
    d_min = 0.0

    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        ax_scatter.scatter(
            sub["residual_error"], sub["gate_value"],
            c=REGIME_COLORS[regime],
            label=REGIME_LABELS[regime],
            alpha=0.45, s=18, edgecolors="none",
            zorder=2,
        )

    # Theoretical sigmoid
    d_vals = np.linspace(d_min, d_max, 500)
    g_vals = theoretical_gate(d_vals)
    ax_scatter.plot(
        d_vals, g_vals,
        color="black", linewidth=2.0, linestyle="--",
        label=rf"$g = \sigma({int(GATE_ALPHA)}(\tau - D_{{res}}))$",
        zorder=3,
    )

    # Threshold line
    ax_scatter.axvline(GATE_TAU, color="grey", linewidth=1.0, linestyle=":",
                       label=f"$\\tau = {GATE_TAU}$ (threshold)")
    ax_scatter.axhline(0.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

    ax_scatter.set_xlabel("Residual error $D_{res}$ (support MSE)", fontsize=11)
    ax_scatter.set_ylabel("Gate value $g$", fontsize=11)
    ax_scatter.set_title("Gate Calibration", fontweight="bold", fontsize=12)
    ax_scatter.set_xlim(d_min, d_max)
    ax_scatter.set_ylim(-0.02, 1.05)
    ax_scatter.legend(fontsize=8, framealpha=0.85)
    ax_scatter.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ---- Right: distribution of g by regime ----
    regime_present = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    palette = {r: REGIME_COLORS[r] for r in regime_present}

    sns.violinplot(
        data=df[df["regime"].isin(regime_present)],
        x="regime", y="gate_value",
        hue="regime",
        order=regime_present,
        palette=palette,
        inner="box",
        ax=ax_violin,
        linewidth=0.8,
        legend=False,
    )
    ax_violin.set_xlabel("Regime", fontsize=11)
    ax_violin.set_ylabel("Gate value $g$", fontsize=11)
    ax_violin.set_title("Gate Distribution by Regime", fontweight="bold", fontsize=12)
    ax_violin.set_ylim(-0.02, 1.05)
    ax_violin.set_xticks(range(len(regime_present)))
    ax_violin.set_xticklabels(
        [REGIME_LABELS.get(r, r) for r in regime_present], rotation=10, ha="right"
    )
    ax_violin.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate median gate values
    for i, regime in enumerate(regime_present):
        med = df[df["regime"] == regime]["gate_value"].median()
        ax_violin.text(i, med + 0.04, f"med={med:.2f}",
                       ha="center", fontsize=7.5, color="black")

    # ---- Shared style ----
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Safety Gate Calibration — Model C\n"
        rf"$g = \sigma({int(GATE_ALPHA)} \cdot (\tau - D_{{res}}))$, "
        rf"$\tau = {GATE_TAU}$",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_MAIN, dpi=150, bbox_inches="tight")
    print(f"✅  Main calibration plot saved to {OUT_MAIN}")
    plt.close()


# ---------------------------------------------------------------------------
# Faceted plot: one panel per steps_available
# ---------------------------------------------------------------------------

def plot_by_steps(df: pd.DataFrame):
    """
    Shows whether the gate calibration curve is consistent across different
    numbers of support steps. If the curve shifts right as steps increase,
    adaptation improves and the gate opens more reliably.
    """
    if "steps_available" not in df.columns:
        print("  ⚠️  steps_available column not found — skipping faceted plot.")
        return

    steps = sorted(df["steps_available"].unique())
    n_cols = min(len(steps), 4)
    n_rows = (len(steps) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             sharey=True)
    axes_flat = np.array(axes).flatten()

    d_max_global = df["residual_error"].quantile(0.99) * 1.05
    d_vals = np.linspace(0, d_max_global, 500)
    g_theory = theoretical_gate(d_vals)

    for ax_idx, steps_val in enumerate(steps):
        ax = axes_flat[ax_idx]
        sub = df[df["steps_available"] == steps_val]

        for regime in REGIME_ORDER:
            r_sub = sub[sub["regime"] == regime]
            if r_sub.empty:
                continue
            ax.scatter(
                r_sub["residual_error"], r_sub["gate_value"],
                c=REGIME_COLORS[regime], alpha=0.5, s=12, edgecolors="none",
            )

        ax.plot(d_vals, g_theory, color="black", linewidth=1.5,
                linestyle="--", alpha=0.7)
        ax.axvline(GATE_TAU, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(f"steps = {int(steps_val)}", fontsize=9)
        ax.set_xlim(0, d_max_global)
        ax.set_ylim(-0.02, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("$g$", fontsize=9)
        ax.set_xlabel("$D_{res}$", fontsize=8)

    # Hide unused axes
    for ax in axes_flat[len(steps):]:
        ax.set_visible(False)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=REGIME_COLORS[r], markersize=7,
                   label=REGIME_LABELS[r])
        for r in REGIME_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="--",
                   label="Theoretical $g$")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Gate Calibration by Support Steps", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_FACET, dpi=150, bbox_inches="tight")
    print(f"✅  Faceted plot saved to {OUT_FACET}")
    plt.close()


# ---------------------------------------------------------------------------
# Diagnostics printed to stdout
# ---------------------------------------------------------------------------

def print_diagnostics(df: pd.DataFrame):
    print("\n── Gate Calibration Diagnostics ──")
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        # Verify empirical gate ≈ theoretical gate (should be ~1.0)
        g_theory = theoretical_gate(sub["residual_error"].values)
        corr = np.corrcoef(sub["gate_value"].values, g_theory)[0, 1]
        print(
            f"  {regime}  n={len(sub):4d}  "
            f"gate mean={sub['gate_value'].mean():.3f}  "
            f"D_res mean={sub['residual_error'].mean():.4f}  "
            f"corr(empirical, theoretical)={corr:.6f}"
        )

    # Fraction of tasks where gate < 0.5 (model fell back to prior)
    print("\n  Fraction of tasks with g < 0.5 (gate mostly closed):")
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        frac = (sub["gate_value"] < 0.5).mean()
        print(f"    {regime}: {frac:.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("📐 Gate Calibration Plot")
    print("=" * 50)

    df = load_results()
    print_diagnostics(df)
    plot_main(df)
    plot_by_steps(df)
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
