#glucose/generate_simple.py
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from glucoenv.T1DEnv import make

# --- CONFIG ---
OUTPUT_DIR = "data/glucose_simple"
STEPS = 201
N_TRAIN = 8
N_TEST = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Running Simulation on: {DEVICE.upper()}")

def get_patient_id(i):
    return f"adult#{i+1:03d}"

def safe_convert(obs_in):
    """Flatten and detach tensors/tuples."""
    if isinstance(obs_in, tuple):
        obs_in = obs_in[0]
    if isinstance(obs_in, torch.Tensor):
        obs_in = obs_in.detach().cpu().numpy()
    return np.array(obs_in, dtype=np.float32).flatten()

def generate_trajectory(patient_name, seed):
    try:
        env = make(env=[patient_name], n_env=1, scenario="moderate", 
                   device=DEVICE, seed=seed, env_type="test", obs_type="current")
    except:
        return None

    # 1. Get Initial State
    raw_reset = env.reset()
    obs_0 = safe_convert(raw_reset)

    traj = []
    
    # 2. Run Steps to determine "True" dimension
    for _ in range(STEPS - 1):
        action = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
        raw_step = env.step(action)
        
        # Parse Step
        if isinstance(raw_step, tuple):
            done = raw_step[2] if len(raw_step) >= 3 else False
        else:
            done = False
            
        obs_t = safe_convert(raw_step)
        traj.append(obs_t)
        
        if done: break
            
    # 3. FIX THE MISMATCH (The Magic Fix)
    # The step loop defines the 'true' dimension (e.g., 1D)
    # We slice obs_0 to match obs_t
    step_dim = traj[0].shape[0]
    if obs_0.shape[0] != step_dim:
        # Assuming BG is the first index (standard in T1D models)
        obs_0 = obs_0[:step_dim]

    # Prepend the fixed initial state
    traj.insert(0, obs_0)

    # 4. Stack and Save
    full_traj = np.stack(traj, axis=0)
    
    if len(full_traj) < STEPS:
        pad = np.tile(full_traj[-1], (STEPS - len(full_traj), 1))
        full_traj = np.concatenate([full_traj, pad], axis=0)
        
    return full_traj[:STEPS]

def save_split(split, patients, roles, save_dir):
    metadata = []
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"\n🔹 Processing {split} ({len(patients)} patients)...")
    role_s, role_q = roles
    
    for i, patient in enumerate(patients):
        for run in tqdm(range(10), desc=f"   {patient}", leave=True):
            
            # Support
            s_path = os.path.join(split_dir, f"{patient}_run{run}_support.npy")
            if not os.path.exists(s_path):
                traj_s = generate_trajectory(patient, seed=i*100 + run)
                if traj_s is not None:
                    np.save(s_path, traj_s)
                    metadata.append({
                        "theta_id": patient, "split": split, "role": role_s, 
                        "file_path": s_path, "dim": traj_s.shape[-1]
                    })
            
            # Query
            q_path = os.path.join(split_dir, f"{patient}_run{run}_query.npy")
            if not os.path.exists(q_path):
                traj_q = generate_trajectory(patient, seed=i*100 + run + 5000)
                if traj_q is not None:
                    np.save(q_path, traj_q)
                    metadata.append({
                        "theta_id": patient, "split": split, "role": role_q, 
                        "file_path": q_path, "dim": traj_q.shape[-1]
                    })
            
    return pd.DataFrame(metadata)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_patients = [get_patient_id(i) for i in range(N_TRAIN)]
    df_train = save_split("train", train_patients, ("train_inner", "train_outer"), OUTPUT_DIR)
    
    test_patients = [get_patient_id(i) for i in range(N_TRAIN, N_TRAIN + N_TEST)]
    df_test = save_split("test", test_patients, ("support", "query"), OUTPUT_DIR)
    
    df_val = df_test.copy()
    df_val['split'] = 'val'
    df_val['role'] = 'val'
    
    full_index = pd.concat([df_train, df_test, df_val], ignore_index=True)
    full_index.to_csv(os.path.join(OUTPUT_DIR, "index.csv"), index=False)
    print(f"\n✅ Clean Glucose Data Saved: {OUTPUT_DIR}")