# baselines/train_transfer_weak.py
# Weak transfer baseline: NO ENCODER. Single global Neural SDE + Head trained
# with z = 0 on all training data. This represents a "one average physics"
# model that will later be warm-started on each new task.

import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch


def train_transfer_weak():
    device = torch.device(cfg.device)
    print("🔥 Weak Transfer Baseline: Global SDE with z=0 (NO ENCODER)")
    print("Strategy: Train a single Neural SDE + Head on all train data with fixed z=0.\n")

    # 1. Load data
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    train_ds = TrajectoryDataset(index_path, "train", "train_inner", check_shapes=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    # 2. Model: SDE + Head ONLY (no encoder)
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    optimizer = optim.Adam(list(sde.parameters()) + list(head.parameters()), lr=1e-3)

    # 3. Training loop
    epochs = 50
    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.global_seed if hasattr(cfg, "global_seed") else 12345)

    print("🚀 Starting weak transfer training (z fixed to 0)...\n")

    for epoch in range(1, epochs + 1):
        sde.train()
        head.train()
        total_loss = 0.0

        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            x0 = batch[:, 0, :]

            # z fixed to zero (NO ENCODER)
            z_zeros = torch.zeros(x0.size(0), z_dim, device=device)

            traj_pred = simulate_neural_sde_batch(sde, x0, z_zeros, T, n_steps, x_max, gen)

            loss_path = F.mse_loss(traj_pred, batch)
            loss_head = F.mse_loss(head(traj_pred[:, -1, :], z_zeros), batch[:, -1, :])
            loss = loss_path + 0.1 * loss_head

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sde.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {"sde": sde.state_dict(), "head": head.state_dict()},
        "checkpoints/transfer_weak_no_encoder.pt",
    )
    print("\n✅ Saved weak transfer baseline to checkpoints/transfer_weak_no_encoder.pt")


if __name__ == "__main__":
    train_transfer_weak()
