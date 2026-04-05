import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURATION ---
# We point specifically to your CLEAN file for Model C
FILES = {
    "Model C (Ours)": "results/gated_regularized_final_fixed.csv",  # <--- USES THE CLEAN FILE
    "GRU Baseline":   "results/gru_baseline_sweep.csv",
    "MAML":           "results/maml_results_full.csv",
    "Scratch":        "results/scratch_sweep_results_full.csv",
    "Weak Transfer":  "results/transfer_weak_results_full.csv"
}

def load_and_standardize(name, path):
    if not os.path.exists(path):
        print(f"⚠️  Skipping {name}: File not found ({path})")
        return None
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {name}: {e}")
        return None
    
    # Standardize column names to 'mse' and 'steps'
    # Your file uses 'mse_rollout' and 'steps_available'
    cols_map = {
        "MSE_Rollout": "mse", 
        "mse_rollout": "mse",
        "MSE": "mse",
        "steps_available": "steps",
        "Steps": "steps",
        "regime": "regime",
        "Regime": "regime"
    }
    df = df.rename(columns=cols_map)
    
    # Filter for essential columns
    if 'mse' not in df.columns or 'steps' not in df.columns:
        print(f"⚠️  Skipping {name}: Missing 'mse' or 'steps'. Found: {list(df.columns)}")
        return None

    # Force numeric (Critical Safety Step)
    df['mse'] = pd.to_numeric(df['mse'], errors='coerce')
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
    
    # Drop any rows that failed conversion (removes garbage headers)
    df = df.dropna(subset=['mse', 'steps'])
    
    df["Model"] = name
    return df[["regime", "steps", "mse", "Model"]]

def main():
    print("📊 Generating Final Comparison Table...")
    
    all_data = []
    for model_name, file_path in FILES.items():
        df = load_and_standardize(model_name, file_path)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        print("❌ No valid data found!")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # 2. CREATE PIVOT TABLE
    # Group by Regime, Steps, and Model -> Calculate Mean MSE
    pivot = full_df.groupby(['regime', 'steps', 'Model'])['mse'].mean().unstack()
    
    # Reorder columns to put Model C first
    cols = [c for c in pivot.columns if "Model C" in c] + [c for c in pivot.columns if "Model C" not in c]
    pivot = pivot[cols]

    print("\n" + "="*80)
    print("🏆 FINAL MSE COMPARISON TABLE (Lower is Better)")
    print("="*80)
    print(pivot)
    
    pivot.to_csv("results/final_mse_comparison_table.csv")
    print("\n✅ Table saved to: results/final_mse_comparison_table.csv")

    # 3. GENERATE PLOTS
    try:
        plot_comparison(full_df)
    except Exception as e:
        print(f"❌ Plotting Error: {e}")

def plot_comparison(df):
    sns.set_style("whitegrid")
    regimes = [r for r in ['testA', 'testB', 'testC'] if r in df['regime'].unique()]
    
    if not regimes: return

    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5))
    if len(regimes) == 1: axes = [axes]
    
    for i, regime in enumerate(regimes):
        ax = axes[i]
        subset = df[df['regime'] == regime]
        
        sns.lineplot(
            data=subset, x="steps", y="mse", 
            hue="Model", style="Model", 
            markers=True, dashes=False, ax=ax, linewidth=2.5
        )
        
        ax.set_title(f"Regime: {regime}", fontweight='bold')
        ax.set_xlabel("Support Steps")
        ax.set_ylabel("MSE (Log Scale)")
        ax.set_yscale("log")
        
        if i == 0: 
            ax.legend(loc='upper right')
        else:
            if ax.get_legend(): ax.get_legend().remove()
            
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/final_comparison_plot.png", dpi=300)
    print("✅ Plot saved to: results/final_comparison_plot.png")

if __name__ == "__main__":
    main()