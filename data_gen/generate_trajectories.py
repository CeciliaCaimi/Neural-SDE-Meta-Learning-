#Changes: Uses parallel batching logic + retries. Handles file IO.
# data_gen/generate_trajectories.py

import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# --- CONFIG & MODULES ---
from config.base_config import cfg
from sde_basis.parameters import Theta
from sde_basis.covariance import sample_correlation
from data_gen.simulate_true_sde import simulate_batch

def generate_dataset():
    print(f"Starting Data Generation on device: {cfg.device}")
    
    # 1. Load Parameters
    if not os.path.exists(cfg.paths.meta_params_path):
        raise FileNotFoundError(f"Meta-params not found at {cfg.paths.meta_params_path}. Run generate_meta_params.py first!")
    
    # Load onto CPU first
    meta_params = torch.load(cfg.paths.meta_params_path, map_location="cpu", weights_only=False)
    
    # 2. Setup Indexing
    index_rows = []
    
    # Generator for simulation noise
    sim_gen = torch.Generator(device=cfg.device)
    sim_gen.manual_seed(cfg.global_seed + 9999) # Using cfg.global_seed

    # 3. Main Loop over Splits
    for split_name, theta_list in meta_params.items():
        print(f"Processing split: {split_name} ({len(theta_list)} thetas)")
        
        # Determine output folder
        if "train" in split_name:
            root_dir = cfg.paths.train_traj_root
        elif "val" in split_name:
            root_dir = cfg.paths.val_traj_root
        else:
            root_dir = cfg.paths.test_traj_root
            
        for theta in tqdm(theta_list, desc=f"Simulating {split_name}"):
            # Move theta to GPU for simulation
            theta = theta.to(cfg.device)
            theta_id = theta.id
            
            # Create Theta Folder
            theta_dir = os.path.join(root_dir, theta_id)
            os.makedirs(theta_dir, exist_ok=True)
            
            # --- Determine required counts based on split (Using cfg.dataset_sizes) ---
            tasks: List[Tuple[str, int]] = []
            if split_name == "train":
                tasks.append(("train_inner", cfg.dataset_sizes.n_train_traj_per_theta_train_inner))
                tasks.append(("val_inner", cfg.dataset_sizes.n_train_traj_per_theta_val_inner))
            elif split_name == "val":
                tasks.append(("val", cfg.dataset_sizes.n_val_traj_per_theta))
            else: # testA, testB, testC
                tasks.append(("support", cfg.dataset_sizes.n_test_support_traj_per_theta))
                tasks.append(("query", cfg.dataset_sizes.n_test_query_traj_per_theta))

            # --- Sample Covariance L (One per Theta) ---
            theta_seed = int(hash(theta_id) % 1e9)
            cov_gen = torch.Generator(device=cfg.device)
            cov_gen.manual_seed(theta_seed)
            
            # Use cfg.basis.x_dim
            _, L = sample_correlation(cfg.basis.x_dim, cov_gen, cfg.device)
            
            # --- Execute Simulation Tasks ---
            for role, count in tasks:
                if count == 0: continue
                
                valid_trajs_list = []
                attempts = 0
                max_retries = cfg.stability.max_retries_per_theta
                
                while len(valid_trajs_list) < count and attempts < max_retries:
                    needed = count - len(valid_trajs_list)
                    batch_size = min(needed * 2, 200) 
                    batch_size = max(batch_size, needed)
                    
                    # Initial condition: Using cfg.init (Restored config section)
                    x0 = torch.randn(batch_size, cfg.basis.x_dim, device=cfg.device) 
                    x0 = x0 * cfg.init.x0_std + cfg.init.x0_mean
                    
                    # RUN SIMULATOR
                    batch_trajs, mask = simulate_batch(
                        theta=theta, 
                        L=L, 
                        x0=x0, 
                        T=cfg.time_grid.T, 
                        n_steps=cfg.time_grid.n_steps, 
                        x_max_abs=cfg.stability.max_state_abs,
                        generator=sim_gen
                    )
                    
                    # Filter valid trajectories
                    good_ones = batch_trajs[mask]
                    
                    for t_idx in range(len(good_ones)):
                        valid_trajs_list.append(good_ones[t_idx].cpu())
                        if len(valid_trajs_list) >= count: break
                    
                    attempts += 1
                
                # Check failure
                if len(valid_trajs_list) < count:
                    print(f"WARNING: Theta {theta_id} ({role}) unstable. Got {len(valid_trajs_list)}/{count}")
                    
                # Save Files
                role_dir = os.path.join(theta_dir, role)
                os.makedirs(role_dir, exist_ok=True)
                
                for k, traj_tensor in enumerate(valid_trajs_list):
                    fname = f"traj_{k}.npy"
                    fpath = os.path.join(role_dir, fname)
                    np.save(fpath, traj_tensor.numpy())
                    
                    # Log index
                    index_rows.append({
                        "split": split_name,
                        "theta_id": theta_id,
                        "role": role,
                        "file_path": fpath
                    })

    # 4. Save Index
    os.makedirs(cfg.paths.data_root, exist_ok=True)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    df = pd.DataFrame(index_rows)
    df.to_csv(index_path, index=False)
    print(f"Done! Generated {len(df)} trajectories. Index saved to {index_path}")

if __name__ == "__main__":
    generate_dataset()