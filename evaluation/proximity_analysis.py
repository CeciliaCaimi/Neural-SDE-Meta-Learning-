#evaluation/proximity_analysis.py
import torch
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from config.base_config import cfg
# We need the Theta class definition to load the pickle file correctly
from sde_basis.parameters import Theta

def flatten_theta(theta_obj) -> torch.Tensor:
    """Flattens theta_b and theta_sigma into a single vector."""
    # Ensure we are accessing attributes correctly depending on how Theta is defined
    b = theta_obj.theta_b.flatten()
    s = theta_obj.theta_sigma.flatten()
    return torch.cat([b, s])

def main():
    print("🔍 Starting Proximity Analysis...")
    
    # 1. Load Meta-Parameters (Ground Truth)
    if not os.path.exists(cfg.paths.meta_params_path):
        raise FileNotFoundError(f"Meta-params not found at {cfg.paths.meta_params_path}")
    
    meta_params = torch.load(cfg.paths.meta_params_path, map_location="cpu")
    
    # 2. Build Training Distribution Matrix
    # This matrix represents the "Known Physics" manifold
    train_thetas = meta_params['train']
    train_vecs = [flatten_theta(t) for t in train_thetas]
    train_matrix = torch.stack(train_vecs) # Shape: (N_train, P)
    print(f"Training distribution shape: {train_matrix.shape}")
    
    # 3. Load Adaptation Results
    res_path = "results/adaptation_results.csv"
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"{res_path} not found. Run few_shot_adapt.py first.")
        
    df = pd.read_csv(res_path)
    
    # 4. Compute Distances for each test task
    distances = []
    
    # Helper to find theta object by ID
    # We combine all test lists to search easily
    all_test_thetas = meta_params.get('testA', []) + \
                      meta_params.get('testB', []) + \
                      meta_params.get('testC', [])
    
    theta_map = {t.id: t for t in all_test_thetas}
    
    print("Computing distances to training manifold...")
    
    for idx, row in df.iterrows():
        theta_id = row['theta_id']
        
        if theta_id not in theta_map:
            # Should not happen if data is consistent
            distances.append(np.nan)
            continue
            
        target_theta = theta_map[theta_id]
        target_vec = flatten_theta(target_theta).unsqueeze(0) # (1, P)
        
        # Calculate Euclidean distance to ALL training tasks
        dists = torch.norm(train_matrix - target_vec, dim=1)
        
        # The "Proximity" is the distance to the NEAREST training task
        min_dist = dists.min().item()
        distances.append(min_dist)
        
    df['proximity_score'] = distances
    
    # 5. Save Analysis
    out_path = "results/adaptation_with_distances.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Analysis saved to {out_path}")
    
    # 6. Print Correlations
    print("\n📊 Correlation: Distance vs Zero-Shot Error")
    print("(Positive correlation = Farther tasks are harder to predict)")
    print("-" * 50)
    
    for regime in ['testA', 'testB', 'testC']:
        sub = df[df['regime'] == regime]
        if sub.empty: continue
        
        # Correlation between Distance and Zero-Shot Error
        corr = sub['proximity_score'].corr(sub['mse_path_zeroshot'])
        
        print(f"[{regime}]")
        print(f"   Correlation: {corr:.4f}")
        print(f"   Avg Distance: {sub['proximity_score'].mean():.4f}")
        print(f"   Avg MSE:      {sub['mse_path_zeroshot'].mean():.4f}")

if __name__ == "__main__":
    main()