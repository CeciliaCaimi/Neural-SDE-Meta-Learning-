import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SETTINGS ---
# Now we run for ALL regimes
REGIMES = ["testA", "testB", "testC"]
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Map: CSV Filename -> (Legend Label, Column Name, Color, Line Style)
FILES = {
    "gated_regularized_final_fixed.csv": ("Latent Meta-SDE (Ours)", "mse_rollout", "#d62728", "-"),
    "maml_results_full.csv":             ("MAML (Gradient-Adapt)", "mse_head_maml", "#ff7f0e", "--"),
    "scratch_sweep_results_full.csv":    ("Random Initialization", "mse_head_scratch", "#7f7f7f", ":"),
    "transfer_weak_results_full.csv":    ("Pretrained Mean (Head-Only)", "mse_head_transfer_weak", "#1f77b4", "-."),
}
def load_data(regime):
    all_data = []
    
    print(f"\n--- Loading Data for {regime} ---")
    
    for filename, (label, col, color, style) in FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        
        if not os.path.exists(path):
            # Fail silently for Strong Transfer if you didn't run it, but warn for others
            if "strong" not in filename:
                print(f"⚠️  Skipping {label}: File not found at {path}")
            continue
            
        try:
            df = pd.read_csv(path)
            df = df[df["regime"] == regime].copy()
            
            if df.empty:
                print(f"⚠️  Skipping {label}: No data for {regime}")
                continue

            # Group by steps to get the mean curve
            df_grouped = df.groupby("steps_available")[col].mean().reset_index()
            df_grouped["Method"] = label
            df_grouped["MSE"] = df_grouped[col]
            df_grouped["Color"] = color
            df_grouped["Style"] = style
            
            print(f"✅ Loaded {label}")
            all_data.append(df_grouped)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    if not all_data:
        return None, None

    combined_df = pd.concat(all_data)
    
    # Load Zero-Shot Floor
    zs_path = os.path.join(RESULTS_DIR, "zero_shot_transfer_results.csv")
    zs_val = None
    if os.path.exists(zs_path):
        zs_df = pd.read_csv(zs_path)
        filtered = zs_df[zs_df["regime"] == regime]
        if not filtered.empty:
            zs_val = filtered["mse_head_zero_shot"].mean()
            print(f"📉 Zero-Shot Floor: {zs_val:.4f}")
            
    return combined_df, zs_val

def plot_regime(regime):
    df, zs_val = load_data(regime)
    if df is None:
        print(f"Skipping plot for {regime} due to missing data.")
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)

    # Plot Lines
    for name, group in df.groupby("Method"):
        # Look up style from FILES dict
        color = group["Color"].iloc[0]
        style = group["Style"].iloc[0]
        
        plt.plot(
            group["steps_available"], 
            group["MSE"], 
            label=name,
            color=color,
            linestyle=style,
            linewidth=3,
            marker='o',
            markersize=8,
            alpha=0.9
        )

    # Zero-Shot Line
    if zs_val is not None:
        plt.axhline(y=zs_val, color='black', linestyle='-', linewidth=1.5, alpha=0.4, label="Zero-Shot Floor")

    # Styling
    plt.title(f"Data Efficiency ({regime})", fontweight="bold", pad=15)
    plt.xlabel("Available Context Steps", fontweight="bold")
    plt.ylabel("Test MSE (Log Scale)", fontweight="bold")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, fontsize=10, loc="best")
    sns.despine()

    # Save
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"hero_curve_{regime}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"🎉 Saved plot to {out_path}")
    plt.close() # Close figure to free memory

if __name__ == "__main__":
    for r in REGIMES:
        plot_regime(r)