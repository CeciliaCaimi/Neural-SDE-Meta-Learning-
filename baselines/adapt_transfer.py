# baselines/adapt_transfer.py
# Transfer baseline: encoder + SDE FROZEN, ONLY HEAD IS FINE-TUNED.
# This mimics standard deep transfer learning where lower layers are fixed
# and only the final prediction head adapts to each new task.

import os
import copy
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead


ADAPT_STEPS = 50
ADAPT_LR = 1e-2
OBS_LEN = 20
N_SHOTS = 2


def simulate_neural_sde_batch(
    sde: NeuralSDE,
    x0: torch.Tensor,
    z: torch.Tensor,
    T: float,
    n_steps: int,
    x_max_abs: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Euler-Maruyama simulation."""
    device = x0.device
    batch_size, d = x0.shape
    dt = T / n_steps
    sqrt_dt = dt ** 0.5

    traj = torch.zeros(batch_size, n_steps + 1, d, device=device)
    x = x0.clone()
    traj[:, 0, :] = x

    for k in range(n_steps):
        t = torch.tensor(k * dt, device=device)
        b = sde.f(t, x, z)
        G = sde.g(t, x, z)
        dW = torch.randn(batch_size, d, device=device, generator=generator) * sqrt_dt
        noise = torch.bmm(G, dW.unsqueeze(-1)).squeeze(-1)
        x = x + b * dt + noise
        x = torch.clamp(x, -x_max_abs, x_max_abs)
        traj[:, k + 1, :] = x

    return traj


def get_task_data(dataset, theta_id, device):
    """Extracts all trajectories for a specific task ID."""
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    indices = rows.index.tolist()
    data = [dataset[i][0] for i in indices]
    return torch.stack(data).to(device)


def infer_z(encoder, support_trajs, obs_len):
    """Infers latent code from support trajectories (using encoder)."""
    encoder.eval()
    with torch.no_grad():
        obs = support_trajs[:, :obs_len, :]
        z_all = encoder(obs)
        z_mean = z_all.mean(dim=0, keepdim=True)
    return z_mean


def evaluate_model(encoder, sde, head, support_trajs, query_trajs, config, gen, obs_len):
    """Evaluates model on query trajectories. Returns (path_mse, head_mse)."""
    encoder.eval()
    sde.eval()
    head.eval()
    
    T = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max_abs = config.stability.max_state_abs

    with torch.no_grad():
        # Infer z from support
        obs = support_trajs[:, :obs_len, :]
        z = encoder(obs).mean(dim=0, keepdim=True)
        
        # Evaluate on query
        x0 = query_trajs[:, 0, :]
        z_expanded = z.expand(x0.size(0), -1)
        traj_pred = simulate_neural_sde_batch(sde, x0, z_expanded, T, n_steps, x_max_abs, gen)
        x_T_pred = traj_pred[:, -1, :]
        final_pred = head(x_T_pred, z_expanded)
        mse_path = torch.mean((traj_pred - query_trajs) ** 2).item()
        mse_head = torch.mean((final_pred - query_trajs[:, -1, :]) ** 2).item()

    return mse_path, mse_head


def fine_tune_transfer_head_only(encoder, sde, head, support_trajs, config, gen, obs_len):
    """
    Transfer adaptation: FREEZE encoder + SDE, FINE-TUNE ONLY THE HEAD.

    This mimics standard deep transfer practice: the representation (encoder+SDE)
    is treated as a generic feature extractor, and only the task-specific head
    is adapted using the limited support data.
    """
    encoder_ft = copy.deepcopy(encoder)
    sde_ft = copy.deepcopy(sde)
    head_ft = copy.deepcopy(head)

    # Freeze encoder and SDE
    for p in encoder_ft.parameters():
        p.requires_grad = False
    for p in sde_ft.parameters():
        p.requires_grad = False

    # Optimizer ONLY over head parameters
    optimizer = optim.Adam(head_ft.parameters(), lr=ADAPT_LR)

    encoder_ft.eval()   # frozen
    sde_ft.eval()       # frozen
    head_ft.train()     # only this adapts

    T = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max_abs = config.stability.max_state_abs

    x0 = support_trajs[:, 0, :]
    target_final = support_trajs[:, -1, :]
    obs = support_trajs[:, :obs_len, :]

    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()

        # z from frozen encoder
        with torch.no_grad():
            z = encoder_ft(obs).mean(dim=0, keepdim=True)
            z_expanded = z.expand(x0.size(0), -1)
            traj_pred = simulate_neural_sde_batch(
                sde_ft, x0, z_expanded, T, n_steps, x_max_abs, gen
            )
            x_T_pred = traj_pred[:, -1, :]

        # Only head has gradients
        pred = head_ft(x_T_pred, z_expanded)
        loss = F.mse_loss(pred, target_final)

        loss.backward()
        optimizer.step()

    return encoder_ft, sde_ft, head_ft


def main():
    device = torch.device(cfg.device)
    print("\n" + "=" * 80)
    print("🔥 Baseline: Transfer Learning (Encoder + SDE FROZEN, HEAD-ONLY ADAPTATION)")
    print("=" * 80)
    print(f"Config: {N_SHOTS}-Shot | OBS_LEN={OBS_LEN} | Steps={ADAPT_STEPS}")
    print("Strategy: Load transfer model → Infer z with frozen encoder →")
    print("          Fine-tune ONLY the forecasting head per task.")
    print("=" * 80 + "\n")

    ckpt_path = "checkpoints/transfer_epoch_50.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"❌ Transfer checkpoint not found at {ckpt_path}\n"
            f"Run `python -m baselines.train_transfer` first!"
        )

    print(f"📂 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    encoder = TrajEncoder(
        x_dim, z_dim, cfg.latent.encoder_hidden_dim, num_layers=2, dropout=0.1
    ).to(device)
    
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    sde.load_state_dict(ckpt['sde'])
    head.load_state_dict(ckpt['head'])

    print("✅ Models loaded successfully\n")

    gen = torch.Generator(device=device)
    gen.manual_seed(999)

    results_list = []
    regimes = ['testA', 'testB', 'testC']
    index_path = os.path.join(cfg.paths.data_root, "index.csv")

    for regime in regimes:
        print("=" * 80)
        print(f"🎯 Processing {regime}")
        print("=" * 80)

        try:
            ds_support = TrajectoryDataset(index_path, regime, "support", check_shapes=True)
            ds_query = TrajectoryDataset(index_path, regime, "query", check_shapes=True)
        except RuntimeError:
            print(f"⚠️  Skipping {regime} (Empty dataset)")
            continue

        tasks = ds_support.metadata['theta_id'].unique()
        print(f"📊 Found {len(tasks)} tasks\n")

        for theta_id in tqdm(tasks, desc=f"Adapting {regime}"):
            full_support = get_task_data(ds_support, theta_id, device)
            support_trajs = full_support[:N_SHOTS]
            query_trajs = get_task_data(ds_query, theta_id, device)

            # Zero-shot evaluation (no head adaptation)
            zs_mse_path, zs_mse_head = evaluate_model(
                encoder, sde, head, support_trajs, query_trajs, cfg, gen, OBS_LEN
            )

            # Head-only transfer adaptation
            encoder_ft, sde_ft, head_ft = fine_tune_transfer_head_only(
                encoder, sde, head, support_trajs, cfg, gen, OBS_LEN
            )
            
            # Evaluate fine-tuned model
            ft_mse_path, ft_mse_head = evaluate_model(
                encoder_ft, sde_ft, head_ft, support_trajs, query_trajs, cfg, gen, OBS_LEN
            )

            results_list.append({
                "regime": regime,
                "theta_id": theta_id,
                "n_shots": N_SHOTS,
                "mse_path_zeroshot": zs_mse_path,
                "mse_path_transfer": ft_mse_path,
                "mse_head_zeroshot": zs_mse_head,
                "mse_head_transfer": ft_mse_head,
            })

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("results/transfer_head_only_results.csv", index=False)

    print("\n" + "=" * 80)
    print("✅ Transfer Learning (Head-Only) Baseline Complete!")
    print("=" * 80)
    
    print("\n📊 RESULTS SUMMARY")
    print("-" * 80)
    print("\n📈 Average PATH MSE (Full Trajectory Physics):")
    print(df.groupby("regime")[["mse_path_zeroshot", "mse_path_transfer"]].mean())
    
    print("\n📈 Average FINAL-STEP MSE (Forecasting):")
    print(df.groupby("regime")[["mse_head_zeroshot", "mse_head_transfer"]].mean())
    
    print(f"\n💾 Results saved to: results/transfer_head_only_results.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
