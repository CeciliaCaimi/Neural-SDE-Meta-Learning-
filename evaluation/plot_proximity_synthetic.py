import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from config.base_config import cfg

# --- CONFIG ---
RESULTS_PATH = "results/gated_regularized_final.csv"  # Points to your main experiment
META_PARAMS_PATH = "data/meta_params.pt"              # Ground truth SDE parameters
SAVE_DIR = "results/plots_synthetic"

def flatten_theta(theta_dict):
    """Flattens the SDE parameters (b and sigma weights) into a single vector."""
    # Assuming theta is a dictionary or object with theta_b and theta_sigma
    # Adjust attribute access if your Theta object is different
    vecs = []
    if hasattr(theta_dict, 'theta_b'):
        vecs.append(theta_dict.theta_b.flatten())
    if hasattr(theta_dict, 'theta_sigma'):
        vecs.append(theta_dict.theta_sigma.flatten())
    return torch.cat(vecs) if vecs else torch.tensor([])

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("🎨 Generating Synthetic Proximity Plots...")

    # 1. Load Ground Truth Parameters
    if not os.path.exists(META_PARAMS_PATH):
        print(f"❌ Missing meta-params: {META_PARAMS_PATH}")
        return
    
    # Load all parameters
    meta_params = torch.load(META_PARAMS_PATH, map_location="cpu")
    
    # Extract Vectors and Labels
    all_vectors = []
    labels = []
    
    # Process Train (The Manifold)
    train_vecs = [flatten_theta(t) for t in meta_params['train']]
    train_matrix = torch.stack(train_vecs)
    all_vectors.extend(train_vecs)
    labels.extend(['Train'] * len(train_vecs))
    
    # Process Test Sets
    test_data = {}
    for regime in ['testA', 'testB', 'testC']:
        if regime in meta_params:
            vecs = [flatten_theta(t) for t in meta_params[regime]]
            test_data[regime] = (meta_params[regime], torch.stack(vecs))
            all_vectors.extend(vecs)
            labels.extend([regime] * len(vecs))

    # 2. PCA Projection (Visualization)
    print("   Computing PCA...")
    X = torch.stack(all_vectors).numpy()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for Plotting
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Regime'] = labels

    # =======================================================
    # PLOT 1: THE MANIFOLD (PCA)
    # =======================================================
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca, x='PC1', y='PC2', hue='Regime', style='Regime',
        palette={'Train': 'gray', 'testA': 'blue', 'testB': 'orange', 'testC': 'red'},
        s=80, alpha=0.8
    )
    plt.title("Parameter Space Manifold (PCA Projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "manifold_pca.png"))
    print("✅ Saved Manifold PCA Plot")

    # 3. Compute Distances & Merge with Results
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ Missing results file: {RESULTS_PATH}")
        return

    df_res = pd.read_csv(RESULTS_PATH)
    
    # We map theta_id to distance
    dist_map = {}
    
    # Compute Distance to Nearest Training Neighbor
    print("   Computing Proximity Scores...")
    for regime in ['testA', 'testB', 'testC']:
        if regime not in test_data: continue
        
        thetas, matrix = test_data[regime]
        
        # Calculate distances to ALL training points
        # matrix: (N_test, D), train_matrix: (N_train, D)
        dists = torch.cdist(matrix, train_matrix) # (N_test, N_train)
        
        # Min distance for each test task
        min_dists = dists.min(dim=1).values.numpy()
        
        for t, d in zip(thetas, min_dists):
            # Assumes theta objects have an 'id' attribute matching the CSV
            if hasattr(t, 'id'):
                dist_map[t.id] = d
    
    # Map back to results dataframe
    df_res['proximity'] = df_res['theta_id'].map(dist_map)
    
    # Filter out NaNs (if any IDs didn't match)
    df_plot = df_res.dropna(subset=['proximity'])
    
    # Filter for a specific step count (e.g., 50) to avoid overplotting
    df_plot = df_plot[df_plot['steps_available'] == 50]

    # =======================================================
    # PLOT 2: GATE VS PROXIMITY
    # =======================================================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot, x='proximity', y='gate_value', hue='regime',
        palette={'testA': 'blue', 'testB': 'orange', 'testC': 'red'},
        s=60
    )
    plt.axhline(1.0, linestyle=':', color='gray')
    plt.axhline(0.0, linestyle=':', color='gray')
    plt.title("Safety Mechanism: Gate vs. Distance from Prior")
    plt.xlabel("Distance to Training Manifold")
    plt.ylabel("Gate Openness (g)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "gate_vs_proximity.png"))
    print("✅ Saved Gate Response Plot")

    # =======================================================
    # PLOT 3: ERROR VS PROXIMITY
    # =======================================================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot, x='proximity', y='mse_rollout', hue='regime',
        palette={'testA': 'blue', 'testB': 'orange', 'testC': 'red'},
        s=60
    )
    plt.yscale('log')
    plt.title("Performance Frontier: Error vs. Difficulty")
    plt.xlabel("Distance to Training Manifold")
    plt.ylabel("MSE Rollout (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "error_vs_proximity.png"))
    print("✅ Saved Error Plot")

if __name__ == "__main__":
    main()