# simulators/blood_glucose_sim.py
import numpy as np
import torch
import os
import pandas as pd
from glucoenv.T1DEnv import make

def run_single_patient(
    patient_name="adult#001",
    seed=0,
    horizon=288,
    scenario_name="moderate",
    device="cpu",
):
    env = make(
        env=[patient_name],
        n_env=1,
        scenario=scenario_name,
        device=device,
        seed=seed,
        env_type="test",
        obs_type="current",
    )

    obs = env.reset()
    traj = []

    for t in range(horizon):
        # GluCoEnv expects flat torch tensor for insulin bolus (shape (n_env,))
        # 0.0 means no bolus (basal handled internally)
        action = torch.tensor([0.0], dtype=torch.float32, device=device)
        obs, reward, done, info = env.step(action)
        traj.append(obs)
        if done:
            break

    traj = np.stack(traj, axis=0)
    return traj

def generate_dataset(n_patients=10, horizon=288, n_seeds=5, out_dir="data/glucose"):
    os.makedirs(out_dir, exist_ok=True)

    results = []
    patient_names = [
        "adult#001", "adult#002", "adult#003", "adult#004", "adult#005",
        "child#001", "child#002", "child#003", "child#004", "child#005",
    ][:n_patients]

    for patient_name in patient_names:
        for seed in range(n_seeds):
            traj = run_single_patient(patient_name=patient_name, seed=seed, horizon=horizon)

            # Save raw trajectory; adjust columns to match your pipeline later
            df_traj = pd.DataFrame(traj.squeeze(-1), columns=["glucose"])
            traj_path = os.path.join(out_dir, f"{patient_name}_seed{seed}.csv")
            df_traj.to_csv(traj_path, index=False)

            results.append({
                "patient_name": patient_name,
                "seed": seed,
                "traj_path": traj_path,
                "theta_id": f"{patient_name}_s{seed}",
                "regime": "glucose",
                "split": "train_inner",
            })

    index_path = os.path.join(out_dir, "index.csv")
    pd.DataFrame(results).to_csv(index_path, index=False)
    print(f"Generated {len(results)} trajectories, index at {index_path}")

if __name__ == "__main__":
    # Quick test
    traj = run_single_patient()
    print("Trajectory shape:", traj.shape)

    # Generate dataset
    generate_dataset(n_patients=10, horizon=288, n_seeds=5, out_dir="data/glucose")
