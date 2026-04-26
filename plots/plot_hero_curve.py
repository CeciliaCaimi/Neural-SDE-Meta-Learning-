import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SETTINGS ---
REGIMES     = ["testA", "testB", "testC"]
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

# Map: CSV filename -> (legend label, metric column, hex color, matplotlib linestyle)
# All current baselines write 'mse_rollout' as the primary metric column.
FILES = {
    "gated_regularized_final_fixed.csv": ("Latent Meta-SDE (Ours)",      "mse_rollout", "#d62728", "-"),
    "maml_results_full.csv":             ("MAML (Gradient-Adapt)",        "mse_rollout", "#ff7f0e", "--"),
    "scratch_sweep_results_full.csv":    ("Random Initialization",        "mse_rollout", "#7f7f7f", ":"),
    "transfer_weak_results_full.csv":    ("Pretrained Mean (Head-Only)",  "mse_rollout", "#1f77b4", "-."),
    "persistence_results_full.csv":      ("Persistence (No-Adapt Floor)", "mse_rollout", "#9467bd", "--"),
}

# Module-level lookups so plot_regime doesn't have to re-derive them from the DataFrame
_COLOR = {meta[0]: meta[2] for meta in FILES.values()}
_STYLE = {meta[0]: meta[3] for meta in FILES.values()}


def load_data(regime):
    """
    Return raw per-task rows (un-aggregated) for all available methods.

    CI computation is deferred to plot_regime so every task contributes an
    independent data point to the confidence interval.  Pre-aggregating here
    (as the old code did) would reduce each method to a single point per step
    and make CI estimation impossible.
    """
    all_data = []
    print(f"\n--- Loading Data for {regime} ---")

    for filename, (label, col, _color, _style) in FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Skipping {label}: File not found at {path}")
            continue
        try:
            df = pd.read_csv(path)
            df = df[df["regime"] == regime].copy()
            if df.empty:
                print(f"⚠️  Skipping {label}: No data for {regime}")
                continue
            if col not in df.columns:
                print(f"⚠️  Skipping {label}: Column '{col}' not found (available: {list(df.columns)})")
                continue
            rows = df[["steps_available", col]].rename(columns={col: "MSE"})
            rows["Method"] = label
            all_data.append(rows)
            n_tasks = rows["steps_available"].value_counts().max()
            print(f"✅ Loaded {label} ({len(rows)} rows, ~{n_tasks} tasks/step)")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    if not all_data:
        return None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Zero-shot scalar reference (horizontal floor line, not a sweep)
    zs_path = os.path.join(RESULTS_DIR, "zero_shot_transfer_results.csv")
    zs_val  = None
    if os.path.exists(zs_path):
        zs_df    = pd.read_csv(zs_path)
        filtered = zs_df[zs_df["regime"] == regime]
        if not filtered.empty and "mse_head_zero_shot" in filtered.columns:
            zs_val = filtered["mse_head_zero_shot"].mean()
            print(f"📉 Zero-Shot Floor: {zs_val:.4f}")

    return combined_df, zs_val


def plot_regime(regime):
    df, zs_val = load_data(regime)
    if df is None:
        print(f"Skipping plot for {regime} due to missing data.")
        return

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Preserve the insertion order from FILES so legend order is deterministic
    methods_ordered = [m[0] for m in FILES.values() if m[0] in df["Method"].unique()]

    for method in methods_ordered:
        grp   = df[df["Method"] == method]
        color = _COLOR[method]
        style = _STYLE[method]

        # Aggregate across tasks at each step count
        stats     = grp.groupby("steps_available")["MSE"].agg(["mean", "std", "count"])
        se        = stats["std"] / np.sqrt(stats["count"])
        ci95_half = 1.96 * se

        # Clamp lower band away from zero so the log-scale fill renders cleanly
        lower = np.maximum(stats["mean"] - ci95_half, stats["mean"] * 1e-3)
        upper = stats["mean"] + ci95_half

        ax.plot(
            stats.index, stats["mean"],
            color=color, linestyle=style, linewidth=2.5,
            marker="o", markersize=7, label=method, alpha=0.9,
        )
        ax.fill_between(
            stats.index, lower, upper,
            color=color, alpha=0.15,
        )

    if zs_val is not None:
        ax.axhline(y=zs_val, color="black", linestyle="-",
                   linewidth=1.5, alpha=0.4, label="Zero-Shot Floor")

    ax.set_title(
        f"Data Efficiency ({regime}) — mean ± 95% CI across tasks",
        fontweight="bold", pad=15,
    )
    ax.set_xlabel("Available Context Steps", fontweight="bold")
    ax.set_ylabel("Test MSE (Log Scale)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fontsize=10, loc="best")
    sns.despine(ax=ax)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"hero_curve_{regime}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"🎉 Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    for r in REGIMES:
        plot_regime(r)
