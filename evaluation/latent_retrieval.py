"""Latent Geometry Retrieval Consistency — Final Validation Exp 3.

For each test task: encode z_test, find top-k nearest in latent space and
in θ-space, report overlap % and rank correlation.

Usage: cd ~/fresh-run && PYTHONPATH=. python evaluation/latent_retrieval.py
"""
import os, sys, torch, numpy as np, pandas as pd
from scipy import stats
sys.path.append(os.getcwd())

from config.base_config import cfg
from models.encoder import TrajEncoder
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories, apply_scaler_to_trajectories


def flatten_theta(theta):
    return torch.cat([theta.theta_b.flatten(), theta.theta_sigma.flatten()])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load encoder
    ckpt = torch.load('checkpoints/meta_epoch_50.pt', map_location=device, weights_only=False)
    L = cfg.latent
    encoder = TrajEncoder(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.encoder_hidden_dim)
    encoder.load_state_dict(ckpt['encoder']); encoder.eval().to(device)

    # Load meta params
    meta = torch.load('data/meta_params.npz', map_location='cpu', weights_only=False)
    train_thetas = meta['train']

    # Flatten training θ vectors
    train_theta_vecs = torch.stack([flatten_theta(t) for t in train_thetas]).numpy()

    # Encode training tasks
    data_dir = "data/test_trajectories"
    train_dir = "data/train_trajectories"

    # Get training z vectors
    train_z_vecs = []
    train_ids_used = []
    if os.path.exists(train_dir):
        train_task_dirs = sorted(os.listdir(train_dir))[:len(train_thetas)]
        for i, task_id in enumerate(train_task_dirs):
            task_path = f"{train_dir}/{task_id}/train_inner"
            if not os.path.exists(task_path):
                continue
            sup_files = sorted(os.listdir(task_path))[:1]
            if not sup_files:
                continue
            support = torch.from_numpy(np.load(f"{task_path}/{sup_files[0]}")).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z = encoder(support[:, :50]).squeeze(0).cpu().numpy()
            train_z_vecs.append(z)
            train_ids_used.append(i)

    if not train_z_vecs:
        print("No training trajectories found. Cannot compute retrieval.")
        return

    train_z_vecs = np.array(train_z_vecs)
    train_theta_vecs = train_theta_vecs[train_ids_used]

    # Evaluate test tasks
    results = []
    for regime in ['testA', 'testB', 'testC']:
        test_thetas = meta[regime]
        theta_ids = sorted([d for d in os.listdir(data_dir) if d.startswith(regime)])

        for i, theta_id in enumerate(theta_ids):
            if i >= len(test_thetas):
                break
            task_dir = f"{data_dir}/{theta_id}/support"
            sup_files = sorted(os.listdir(task_dir))[:1]
            if not sup_files:
                continue

            support = torch.from_numpy(np.load(f"{task_dir}/{sup_files[0]}")).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z_test = encoder(support[:, :50]).squeeze(0).cpu().numpy()

            theta_test = flatten_theta(test_thetas[i]).numpy()

            # Distances
            z_dists = np.linalg.norm(train_z_vecs - z_test, axis=1)
            theta_dists = np.linalg.norm(train_theta_vecs - theta_test, axis=1)

            for k in [5, 10]:
                topk_z = set(np.argsort(z_dists)[:k])
                topk_theta = set(np.argsort(theta_dists)[:k])
                overlap = len(topk_z & topk_theta) / k * 100

                # Rank correlation over all training tasks
                rho, p = stats.spearmanr(z_dists, theta_dists)

                results.append({
                    'regime': regime, 'theta_id': theta_id, 'k': k,
                    'overlap_pct': overlap, 'spearman_rho': rho, 'p_value': p
                })

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/latent_retrieval.csv', index=False)

    print("=== Latent Geometry Retrieval Consistency ===\n")
    for k in [5, 10]:
        print(f"  k = {k}:")
        print(f"  {'Regime':<8} {'Overlap %':<12} {'Spearman ρ':<12}")
        for regime in ['testA', 'testB', 'testC']:
            sub = df[(df['regime'] == regime) & (df['k'] == k)]
            if len(sub) > 0:
                print(f"  {regime:<8} {sub['overlap_pct'].mean():<12.1f} {sub['spearman_rho'].mean():<12.3f}")
        print()

    print(f"✅ Saved results/latent_retrieval.csv")

if __name__ == '__main__':
    main()
