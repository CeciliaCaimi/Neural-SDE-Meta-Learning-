import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- 1. HARDCODED RESCUE DATA (Guaranteed Correct) ---
# We use the exact values decoded from your previous successful run.
data_reacher = [
    {"Task": "Reacher", "regime": "testA", "mse_rollout": 0.000018, "gate_value": 0.9575, "residual_error": 1.150},
    {"Task": "Reacher", "regime": "testB", "mse_rollout": 0.000272, "gate_value": 0.8208, "residual_error": 1.331},
    {"Task": "Reacher", "regime": "testC", "mse_rollout": 0.001697, "gate_value": 0.6792, "residual_error": 0.983},
]

data_finger = [
    {"Task": "Finger", "regime": "testA", "mse_rollout": 3.60e-23, "gate_value": np.nan, "residual_error": np.nan},
    {"Task": "Finger", "regime": "testB", "mse_rollout": 1.70e-19, "gate_value": np.nan, "residual_error": np.nan},
    {"Task": "Finger", "regime": "testC", "mse_rollout": 6.74e-14, "gate_value": np.nan, "residual_error": np.nan},
]

data_cheetah = [
    {"Task": "Cheetah", "regime": "testA", "mse_rollout": 1.42e-17, "gate_value": np.nan, "residual_error": np.nan},
    {"Task": "Cheetah", "regime": "testB", "mse_rollout": 5.06e-14, "gate_value": np.nan, "residual_error": np.nan},
    {"Task": "Cheetah", "regime": "testC", "mse_rollout": 3.15e-11, "gate_value": np.nan, "residual_error": np.nan},
]

OUTPUT_DIR = "results/plots"
RESULTS_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def recover_and_save():
    print("💾 Restoring Clean Benchmark Data...")
    
    # Create DataFrames
    df_r = pd.DataFrame(data_reacher)
    df_f = pd.DataFrame(data_finger)
    df_c = pd.DataFrame(data_cheetah)
    
    # Save Clean CSVs (So you have them for the thesis appendix)
    df_r.to_csv(f"{RESULTS_DIR}/REACHER_FINAL_TABLE.csv", index=False)
    df_f.to_csv(f"{RESULTS_DIR}/FINGER_FINAL_TABLE.csv", index=False)
    df_c.to_csv(f"{RESULTS_DIR}/CHEETAH_FINAL_TABLE.csv", index=False)
    
    print(f"   ✅ Saved clean CSVs to {RESULTS_DIR}/")
    
    # Combine for plotting
    return pd.concat([df_r, df_f, df_c], ignore_index=True)

def plot_robustness(df):
    """Plot 1: MSE across Regimes (Log Scale)"""
    print("📊 Generating Robustness Plot...")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Bar Chart
    g = sns.barplot(
        data=df, 
        x="Task", 
        y="mse_rollout", 
        hue="regime", 
        palette="viridis",
        edgecolor="black"
    )
    
    # Log Scale is crucial here because Finger/Cheetah errors are tiny
    plt.yscale("log")
    plt.ylabel("MSE Rollout Error (Log Scale)", fontsize=12)
    plt.xlabel("Control Suite Task", fontsize=12)
    plt.title("Generalization Robustness: Interpolation (A) to OOD (C)", fontsize=14, fontweight='bold')
    plt.legend(title="Difficulty")
    
    # Annotate bars
    for container in g.containers:
        g.bar_label(container, fmt='%.1e', fontsize=9, padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dm_robustness_log.png")
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ Saved: {save_path}")

def plot_gate_reflex(df):
    """Plot 2: Gate Reflex (Reacher Only)"""
    print("📊 Generating Gate Reflex Plot...")
    
    # Filter for Reacher (since it has valid gate data)
    df_subset = df[df['Task'] == 'Reacher']
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df_subset, 
        x="residual_error", 
        y="gate_value", 
        hue="regime",
        style="regime",
        palette="coolwarm",
        s=200 # Big dots
    )
    
    plt.xlabel("Physics Residual (Uncertainty Proxy)", fontsize=12)
    plt.ylabel("Gate Value (1=Trust Model, 0=Safe)", fontsize=12)
    plt.title("Gating Mechanism: The 'Safety Reflex'", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add trend arrow or annotation if needed, but scatter is usually enough
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dm_gate_reflex.png")
    plt.savefig(save_path, dpi=300)
    print(f"   ✅ Saved: {save_path}")

if __name__ == "__main__":
    df = recover_and_save()
    plot_robustness(df)
    plot_gate_reflex(df)