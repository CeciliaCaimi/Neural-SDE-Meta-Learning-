#deepmind/generate_all.py
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from deepmind.dm_wrapper import GeneralizedDMControl

# --- CONFIGURATION ---
OUTPUT_ROOT = "data/deepmind"
STEPS = 201            
SUPPORT_SIZE = 10      
N_TRAIN = 50           
N_VAL = 20              # <--- NEW: Dedicated Validation Set
N_TEST = 20

TASKS = {
    "reacher": {"domain": "reacher", "task": "easy"},
    "finger":  {"domain": "finger",  "task": "spin"},
    "cheetah": {"domain": "cheetah", "task": "run"}
}

def generate_dataset_for_task(task_key, config):
    print(f"\n🚀 Generating Benchmark: {task_key.upper()}")
    
    save_dir = os.path.join(OUTPUT_ROOT, task_key)
    os.makedirs(save_dir, exist_ok=True)
    
    env = GeneralizedDMControl(config['domain'], config['task'])
    
    def save_regime(regime_name, n_tasks, m_range, f_range):
        regime_path = os.path.join(save_dir, regime_name)
        os.makedirs(regime_path, exist_ok=True)
        metadata = []
        
        # Determine Role Names
        # Train Split uses 'train_inner'/'train_outer'
        # Val/Test Splits use 'support'/'query'
        if regime_name == "train":
            role_support = "train_inner"
            role_query = "train_outer"
        else:
            role_support = "support"
            role_query = "query"
        
        for i in tqdm(range(n_tasks), desc=regime_name, leave=False):
            task_id = f"task_{i:03d}"
            m_scale = np.random.uniform(*m_range)
            f_scale = np.random.uniform(*f_range)
            
            # 1. Generate Data
            traj_support = np.stack([env.generate_trajectory(STEPS, m_scale, f_scale) for _ in range(SUPPORT_SIZE)])
            traj_query = env.generate_trajectory(STEPS, m_scale, f_scale)[np.newaxis, ...]
            
            s_path = os.path.join(regime_path, f"{task_id}_support.pt")
            q_path = os.path.join(regime_path, f"{task_id}_query.pt")
            
            torch.save(torch.tensor(traj_support, dtype=torch.float32), s_path)
            torch.save(torch.tensor(traj_query, dtype=torch.float32), q_path)
            
            # 2. Append Metadata (LONG FORMAT - The Crash Fix)
            # Row 1: Support
            metadata.append({
                "theta_id": task_id,
                "split": regime_name,      # Column 'split'
                "role": role_support,      # Column 'role'
                "path": s_path,            
                "mass_scale": m_scale,
                "friction_scale": f_scale
            })
            
            # Row 2: Query
            metadata.append({
                "theta_id": task_id,
                "split": regime_name,
                "role": role_query,
                "path": q_path,
                "mass_scale": m_scale,
                "friction_scale": f_scale
            })
            
        return pd.DataFrame(metadata)

    # Generate all 5 regimes
    dfs = []
    
    # 1. Train (Normal Physics)
    dfs.append(save_regime("train", N_TRAIN, (0.8, 1.2), (0.8, 1.2)))

    # 2. Validation (Normal Physics, New Seeds) <--- NEW
    dfs.append(save_regime("val", N_VAL, (0.8, 1.2), (0.8, 1.2)))

    # 3. Test A (In-Distribution Test)
    dfs.append(save_regime("testA", N_TEST, (0.8, 1.2), (0.8, 1.2)))

    # 4. Test B (Extrapolation)
    dfs.append(save_regime("testB", N_TEST, (1.5, 2.0), (0.5, 0.8)))

    # 5. Test C (Chaos)
    dfs.append(save_regime("testC", N_TEST, (2.0, 3.0), (0.1, 0.5)))

    # Save Master Index
    full_index = pd.concat(dfs, ignore_index=True)
    full_index.to_csv(os.path.join(save_dir, "index.csv"), index=False)
    print(f"✅ Saved {task_key} index (Rows: {len(full_index)})")

if __name__ == "__main__":
    for k, v in TASKS.items():
        try: generate_dataset_for_task(k, v)
        except Exception as e: print(f"❌ Failed {k}: {e}")