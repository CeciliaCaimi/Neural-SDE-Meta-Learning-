import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
FILES = {
    "Model C (Ours)": "results/gated_regularized_final_fixed.csv",
    "GRU Baseline":   "results/gru_baseline_sweep.csv",
    "MAML":           "results/maml_results_full.csv",
    "Scratch":        "results/scratch_sweep_results_full.csv",
    "Weak Transfer":  "results/transfer_weak_results_full.csv",
    "Persistence":    "results/persistence_results_full.csv",
}

# Column order for the output table (Model C first, then alphabetical)
_MODEL_ORDER = ["Model C (Ours)"] + sorted(k for k in FILES if k != "Model C (Ours)")


def load_and_standardize(name, path):
    if not os.path.exists(path):
        print(f"⚠️  Skipping {name}: File not found ({path})")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {name}: {e}")
        return None

    # Normalise column names across all baseline CSVs
    cols_map = {
        "MSE_Rollout":    "mse",
        "mse_rollout":    "mse",
        "MSE":            "mse",
        "steps_available": "steps",
        "Steps":          "steps",
        "regime":         "regime",
        "Regime":         "regime",
    }
    df = df.rename(columns=cols_map)

    if "mse" not in df.columns or "steps" not in df.columns:
        print(f"⚠️  Skipping {name}: Missing 'mse' or 'steps'. Found: {list(df.columns)}")
        return None

    df["mse"]   = pd.to_numeric(df["mse"],   errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df = df.dropna(subset=["mse", "steps"])
    df["Model"] = name
    return df[["regime", "steps", "mse", "Model"]]


def main():
    print("📊 Generating Final Comparison Table (mean ± std across tasks)...")

    all_data = []
    for model_name, file_path in FILES.items():
        df = load_and_standardize(model_name, file_path)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ No valid data found!")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # -----------------------------------------------------------------------
    # Aggregate: mean and std across tasks for each (regime, steps, Model)
    # -----------------------------------------------------------------------
    agg = (
        full_df
        .groupby(["regime", "steps", "Model"])["mse"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    # Format each cell as "mean ± std"; fall back to bare mean if std is
    # undefined (only one task contributed, so std is NaN).
    def _fmt(row):
        if pd.notna(row["std"]) and row["std"] > 0:
            return f"{row['mean']:.3f} ± {row['std']:.3f}"
        return f"{row['mean']:.3f}"

    agg["formatted"] = agg.apply(_fmt, axis=1)

    # Build string pivot for human-readable output
    present_models = [m for m in _MODEL_ORDER if m in agg["Model"].unique()]
    str_pivot = (
        agg
        .pivot(index=["regime", "steps"], columns="Model", values="formatted")
        .reindex(columns=present_models)
    )

    print("\n" + "=" * 100)
    print("🏆 FINAL MSE COMPARISON TABLE — mean ± std across tasks  (lower is better)")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 20)
    print(str_pivot.to_string())

    os.makedirs("results", exist_ok=True)

    # Formatted table (strings) — primary human-readable artifact
    str_pivot.to_csv("results/final_mse_comparison_table.csv")
    print("\n✅ Formatted table saved to: results/final_mse_comparison_table.csv")

    # Numeric mean-only table — for downstream scripts that need floats
    num_pivot = (
        agg
        .pivot(index=["regime", "steps"], columns="Model", values="mean")
        .reindex(columns=present_models)
    )
    num_pivot.to_csv("results/final_mse_comparison_table_numeric.csv")
    print("✅ Numeric table saved to:    results/final_mse_comparison_table_numeric.csv")

    try:
        plot_comparison(full_df)
    except Exception as e:
        print(f"❌ Plotting Error: {e}")


def plot_comparison(df):
    """
    Line plot of MSE vs. support steps for each model, per regime.
    sns.lineplot aggregates across tasks automatically and shades the
    95% bootstrap confidence interval around each mean line.
    """
    sns.set_style("whitegrid")
    regimes = [r for r in ["testA", "testB", "testC"] if r in df["regime"].unique()]
    if not regimes:
        return

    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), sharey=False)
    if len(regimes) == 1:
        axes = [axes]

    for i, regime in enumerate(regimes):
        ax     = axes[i]
        subset = df[df["regime"] == regime]

        sns.lineplot(
            data=subset,
            x="steps", y="mse",
            hue="Model", style="Model",
            estimator="mean",
            errorbar=("ci", 95),      # 95% bootstrap CI shaded automatically
            markers=True, dashes=False,
            ax=ax, linewidth=2.5,
        )

        ax.set_title(f"Regime: {regime}\n(mean ± 95% CI across tasks)", fontweight="bold")
        ax.set_xlabel("Support Steps")
        ax.set_ylabel("MSE (Log Scale)")
        ax.set_yscale("log")

        if i == 0:
            ax.legend(loc="upper right", fontsize=8, title="Model")
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    out_path = "results/final_comparison_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
