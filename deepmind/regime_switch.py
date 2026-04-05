#deepmind/regime_switch.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from dataloaders.trajectory_datasets import TrajectoryDataset

# --- CONFIG ---
X_DIM = 10           
HIDDEN_DIM = 128     # Matches your checkpoint
DATA_ROOT = "data"   
INDEX_PATH = os.path.join(DATA_ROOT, "index.csv")
CKPT_PATH = "checkpoints/transfer_epoch_50.pt"

T_SWITCH = 50        
WINDOW = 20
RECOVERY_THRESH = 2.0 

def get_smart_split(index_path):
    """Auto-detects a valid split name from the CSV to avoid crashes."""
    if not os.path.exists(index_path):
        return None, None
    
    df = pd.read_csv(index_path)
    available_splits = df['split'].unique()
    print(f"   ℹ️  Available splits in index: {available_splits}")
    
    # Priority list: try these in order
    for candidate in ['test', 'val', 'validation', 'testA', 'train']:
        if candidate in available_splits:
            # Check if it has 'query' or 'val' roles
            roles = df[df['split'] == candidate]['role'].unique()
            target_role = 'query' if 'query' in roles else roles[0]
            print(f"   ✅ Auto-selected Split: '{candidate}' | Role: '{target_role}'")
            return candidate, target_role
            
    return available_splits[0], df[df['split']==available_splits[0]]['role'].unique()[0]

def get_trajectory(dataset, theta_id):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    if rows.empty: return None
    idx = rows.index.tolist()[0]
    traj = dataset[idx][0]
    return traj[:, :X_DIM] if traj.shape[-1] >= X_DIM else traj

def calc_recovery_metrics(errors, switch_relative_idx, threshold_val):
    pre_switch = errors[:switch_relative_idx]
    post_switch = errors[switch_relative_idx:]
    baseline_mse = np.mean(pre_switch) if len(pre_switch) > 0 else 0.0
    peak_shock = np.max(post_switch)
    target_level = baseline_mse * threshold_val
    recovery_steps = 0
    recovered = False
    for i, err in enumerate(post_switch):
        if err < target_level and i > 5:
            recovery_steps = i
            recovered = True
            break
    if not recovered: recovery_steps = len(post_switch)
    return baseline_mse, peak_shock, recovery_steps

def run_regime_switch():
    print(f"🚀 Running Regime Switch (Smart Load)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint missing: {CKPT_PATH}")
        return

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    z_dim = cfg.latent.latent_dim
    encoder = TrajEncoder(X_DIM, z_dim, HIDDEN_DIM).to(device)
    sde = NeuralSDE(X_DIM, z_dim, HIDDEN_DIM).to(device)
    
    enc_key = 'encoder_state_dict' if 'encoder_state_dict' in checkpoint else 'encoder'
    sde_key = 'sde_state_dict' if 'sde_state_dict' in checkpoint else 'sde'
    
    try:
        encoder.load_state_dict(checkpoint[enc_key])
        sde.load_state_dict(checkpoint[sde_key])
        print("   ✅ Model Loaded.")
    except Exception as e:
        print(f"❌ Weight mismatch: {e}")
        return
    encoder.eval(); sde.eval()

    # 2. Smart Data Load
    split_name, role_name = get_smart_split(INDEX_PATH)
    if split_name is None:
        print("❌ Could not read index.csv")
        return

    ds = TrajectoryDataset(INDEX_PATH, split_name, role_name)
    ids = ds.metadata["theta_id"].unique()
    
    if len(ids) < 2:
        print("❌ Not enough unique tasks found to simulate switch.")
        print(f"   Found IDs: {ids}")
        return

    # Use first and last ID to ensure difference
    traj_A = get_trajectory(ds, ids[0]).to(device)
    traj_C = get_trajectory(ds, ids[-1]).to(device)
    ground_truth = torch.cat([traj_A[:T_SWITCH], traj_C[:T_SWITCH]], dim=0)
    
    # 3. Simulation
    errors, z_norms = [], []
    print("\n" + "="*75)
    print(f"{'Step':<5} | {'Regime':<10} | {'MSE Error':<15} | {'Latent Norm z(t)':<20}")
    print("-" * 75)
    
    for t in range(WINDOW, len(ground_truth)):
        context = ground_truth[t-WINDOW : t].unsqueeze(0)
        with torch.no_grad():
            z = encoder(context)
            if isinstance(z, tuple): z = z[0]
            z_mag = torch.norm(z).item()
            z_norms.append(z_mag)
            
            x_curr = ground_truth[t-1].unsqueeze(0)
            drift = sde.f(0, x_curr, z if z.dim()==2 else z.unsqueeze(0))
            x_pred = x_curr + drift * 0.05
            
            mse = F.mse_loss(x_pred, ground_truth[t].unsqueeze(0)).item()
            errors.append(mse)
            print(f"{t:<5} | {'Normal' if t < T_SWITCH else 'SHOCK':<10} | {mse:.6f}        | {z_mag:.6f}")

    # 4. Metrics & Plot
    base, peak, rec = calc_recovery_metrics(errors, T_SWITCH-WINDOW, RECOVERY_THRESH)
    
    print("="*75)
    print(f"📊 RESULTS (Hidden={HIDDEN_DIM}, Split={split_name}):")
    print(f"   - Peak Shock Error:  {peak:.6f}")
    print(f"   - Time-to-Recover:   {rec} steps")
    print(f"   - Baseline MSE:      {base:.6f}")
    print("="*75)

    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1); plt.plot(errors, 'r'); plt.axvline(x=T_SWITCH-WINDOW, color='k', ls='--'); plt.title("Error Response")
    plt.subplot(2,1,2); plt.plot(z_norms, 'b'); plt.axvline(x=T_SWITCH-WINDOW, color='k', ls='--'); plt.title("Latent Adaptation")
    plt.tight_layout(); plt.savefig("results/plots/regime_switch_10D.png")
    print("✅ Plot saved to results/plots/regime_switch_10D.png")

if __name__ == "__main__":
    run_regime_switch()