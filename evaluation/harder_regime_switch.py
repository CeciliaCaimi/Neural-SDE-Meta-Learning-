"""Harder Regime Switch — Priority 3.1."""
import os, sys, torch, numpy as np, pandas as pd
sys.path.append(os.getcwd())

from config.base_config import cfg
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import gated_inference

def load_models(device):
    ckpt = torch.load('checkpoints/meta_epoch_50.pt', map_location=device, weights_only=False)
    L = cfg.latent
    encoder = TrajEncoder(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim,
                          hidden_dim=L.encoder_hidden_dim)
    encoder.load_state_dict(ckpt['encoder']); encoder.eval().to(device)
    sde = NeuralSDE(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.sde_hidden_dim)
    sde.load_state_dict(ckpt['sde']); sde.eval().to(device)
    head = ForecastHead(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.head_hidden_dim)
    head.load_state_dict(ckpt['head']); head.eval().to(device)
    return encoder, sde, head

def scale_trajectories(trajs, scale_factor):
    mean = trajs.mean(dim=(0, 1), keepdim=True)
    return mean + scale_factor * (trajs - mean)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, sde, head = load_models(device)
    gen = torch.Generator(device=device).manual_seed(42)

    data_dir = "data/test_trajectories"
    theta_ids = sorted([d for d in os.listdir(data_dir) if d.startswith('testC')])[:10]

    results = []
    steps_to_test = [50, 201]

    for scale in [1.0, 1.5, 2.0, 3.0]:
        print(f"\n--- Scale factor: {scale}x ---")
        for theta_id in theta_ids:
            task_dir = f"{data_dir}/{theta_id}"
            sup_files = sorted(os.listdir(f"{task_dir}/support"))
            qry_files = sorted(os.listdir(f"{task_dir}/query"))

            support = torch.stack([torch.from_numpy(np.load(f"{task_dir}/support/{f}")).float()
                                   for f in sup_files[:2]]).to(device)
            query = torch.stack([torch.from_numpy(np.load(f"{task_dir}/query/{f}")).float()
                                 for f in qry_files[:2]]).to(device)

            if scale != 1.0:
                support = scale_trajectories(support, scale)
                query = scale_trajectories(query, scale)

            if torch.isnan(support).any() or support.abs().max() > 100:
                print(f"  {theta_id} diverged at scale {scale}, skipping")
                continue

            for steps in steps_to_test:
                sup_slice = support[:, :steps]
                try:
                    scaler = fit_scaler_on_trajectories(sup_slice)
                    metrics = gated_inference(encoder, sde, head, sup_slice, query, gen, cfg, scaler)
                    results.append({
                        'scale': scale, 'theta_id': theta_id, 'steps': steps,
                        'mse_rollout': metrics['mse_rollout'],
                        'gate_value': metrics['gate_value'],
                        'residual_error': metrics['residual_error']
                    })
                    print(f"  {theta_id} steps={steps}: MSE={metrics['mse_rollout']:.4f} gate={metrics['gate_value']:.3f}")
                except Exception as e:
                    print(f"  {theta_id} steps={steps} failed: {e}")

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/harder_regime_switch.csv', index=False)

    print("\n\n=== Harder Regime Switch Results ===")
    for steps in steps_to_test:
        print(f"\n  Steps = {steps}:")
        print(f"  {'Scale':<8} {'Mean MSE':>10} {'Mean Gate':>10} {'N':>5}")
        for scale in [1.0, 1.5, 2.0, 3.0]:
            sub = df[(df['scale'] == scale) & (df['steps'] == steps)]
            if len(sub) > 0:
                print(f"  {scale:<8} {sub['mse_rollout'].mean():>10.4f} {sub['gate_value'].mean():>10.3f} {len(sub):>5}")

if __name__ == '__main__':
    main()
