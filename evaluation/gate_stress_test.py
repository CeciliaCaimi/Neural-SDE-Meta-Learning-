"""Gate Stress Test under Hard OOD — Final Validation Exp 2.

Scale system parameters θ_test = c * θ for c ∈ {2, 3, 5}.
Evaluate Always ON (g=1), Always OFF (g=0), Adaptive Gate.
Report rollout MSE at N=50 and N=201.

Goal: identify regimes where Always ON degrades and gate prevents failure.

Usage: cd ~/fresh-run && PYTHONPATH=. python evaluation/gate_stress_test.py
"""
import os, sys, copy, torch, numpy as np, pandas as pd
import torch.nn.functional as F
sys.path.append(os.getcwd())

from config.base_config import cfg
from dataloaders.trajectory_datasets import fit_scaler_on_trajectories
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from adaptation.gated_finetuning_regularized import (
    adapt_model, compute_residual, GATE_ALPHA, GATE_TAU, MC_SAMPLES
)
from training.train_meta import simulate_neural_sde_batch


def load_models(device):
    ckpt = torch.load('checkpoints/meta_epoch_50.pt', map_location=device, weights_only=False)
    L = cfg.latent
    encoder = TrajEncoder(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.encoder_hidden_dim)
    encoder.load_state_dict(ckpt['encoder']); encoder.eval().to(device)
    sde = NeuralSDE(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.sde_hidden_dim)
    sde.load_state_dict(ckpt['sde']); sde.eval().to(device)
    head = ForecastHead(x_dim=cfg.basis.x_dim, z_dim=L.latent_dim, hidden_dim=L.head_hidden_dim)
    head.load_state_dict(ckpt['head']); head.eval().to(device)
    return encoder, sde, head


def evaluate_with_gate(encoder, sde, head, support, query, gen, gate_override=None):
    """Run adaptation and evaluate with a specific gate value or adaptive gate."""
    device = support.device
    scaler = fit_scaler_on_trajectories(support)

    from dataloaders.trajectory_datasets import apply_scaler_to_trajectories
    support_in = apply_scaler_to_trajectories(support, scaler)
    query_in = apply_scaler_to_trajectories(query, scaler)

    # Encode + adapt
    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        z_init = encoder(support_in[:, :enc_len]).mean(dim=0, keepdim=True)

    head_opt, z_opt, _ = adapt_model(sde, head, z_init, support_in, gen, cfg)

    # Gate
    if gate_override is not None:
        g = gate_override
    else:
        d_res = compute_residual(sde, head_opt, z_opt, support_in, gen, cfg)
        data_var = support_in.var().item()
        N = support_in.shape[1]
        d_norm = d_res / (data_var * (N ** 0.5) + 1e-8)
        g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_norm))).item()

    # Predict
    B_q = query_in.shape[0]
    z_smart = z_opt.expand(B_q, -1)
    z_safe = torch.zeros_like(z_smart)
    T_full = cfg.time_grid.T; n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs

    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            t_smart = simulate_neural_sde_batch(sde, query_in[:, 0], z_smart, T_full, n_steps, x_max, gen)
            t_safe = simulate_neural_sde_batch(sde, query_in[:, 0], z_safe, T_full, n_steps, x_max, gen)
            mc_preds.append((1 - g) * t_safe + g * t_smart)

    pred = torch.stack(mc_preds).mean(dim=0)
    valid_len = min(pred.shape[1], query_in.shape[1])
    mse = F.mse_loss(pred[:, :valid_len], query_in[:, :valid_len]).item()
    return mse, g


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

    for c in [2, 3, 5]:
        print(f"\n--- Scale c={c} ---")
        for theta_id in theta_ids:
            task_dir = f"{data_dir}/{theta_id}"
            sup_files = sorted(os.listdir(f"{task_dir}/support"))
            qry_files = sorted(os.listdir(f"{task_dir}/query"))

            support = torch.stack([torch.from_numpy(np.load(f"{task_dir}/support/{f}")).float()
                                   for f in sup_files[:2]]).to(device)
            query = torch.stack([torch.from_numpy(np.load(f"{task_dir}/query/{f}")).float()
                                 for f in qry_files[:2]]).to(device)

            support = scale_trajectories(support, c)
            query = scale_trajectories(query, c)

            if torch.isnan(support).any() or support.abs().max() > 100:
                print(f"  {theta_id} diverged, skipping")
                continue

            for steps in [50, 201]:
                sup_slice = support[:, :steps]
                try:
                    mse_on, _ = evaluate_with_gate(encoder, sde, head, sup_slice, query, gen, gate_override=1.0)
                    mse_off, _ = evaluate_with_gate(encoder, sde, head, sup_slice, query, gen, gate_override=0.0)
                    mse_gate, g = evaluate_with_gate(encoder, sde, head, sup_slice, query, gen, gate_override=None)

                    results.append({'scale': c, 'theta_id': theta_id, 'steps': steps,
                                    'mse_always_on': mse_on, 'mse_always_off': mse_off,
                                    'mse_adaptive': mse_gate, 'gate_value': g})
                    print(f"  {theta_id} steps={steps}: ON={mse_on:.3f} OFF={mse_off:.3f} Gate={mse_gate:.3f} (g={g:.3f})")
                except Exception as e:
                    print(f"  {theta_id} steps={steps} failed: {e}")

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/gate_stress_test.csv', index=False)

    print("\n\n=== Gate Stress Test Results ===")
    for steps in [50, 201]:
        print(f"\n  Steps = {steps}:")
        print(f"  {'Scale':<6} {'Always ON':<12} {'Always OFF':<12} {'Adaptive':<12} {'Gate wins?'}")
        for c in [2, 3, 5]:
            sub = df[(df['scale'] == c) & (df['steps'] == steps)]
            if len(sub) > 0:
                on = sub['mse_always_on'].mean()
                off = sub['mse_always_off'].mean()
                ada = sub['mse_adaptive'].mean()
                best = min(on, off, ada)
                winner = "✅ Gate" if ada == best else ("ON" if on == best else "OFF")
                print(f"  {c:<6} {on:<12.4f} {off:<12.4f} {ada:<12.4f} {winner}")

    print(f"\n✅ Saved results/gate_stress_test.csv")

if __name__ == '__main__':
    main()
