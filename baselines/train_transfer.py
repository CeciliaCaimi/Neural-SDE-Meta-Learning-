# baselines/train_transfer.py
# To run: python -m baselines.train_transfer

"""
Baseline 1: Transfer Learning (Warm-Start)

Key Difference from Meta-Learning:
- Meta: Trains on separate tasks, learns task structure via Encoder
- Transfer: Trains on UNION of all data (ignores task boundaries)
  → Learns "average physics" but not task-specific patterns

Theory (Yosinski et al., 2014):
  Good initialization beats random initialization.
  But does it beat structured meta-learning?
"""

import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead


# -----------------------------
# Hyperparameters (Match Meta-Learning for Fair Comparison)
# -----------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 50
OBS_LEN = 50  # Encoder observation length


# -----------------------------
# Simulator (Copied from train_meta.py)
# -----------------------------
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


# -----------------------------
# Training Loop
# -----------------------------
def train_transfer_baseline():
    device = torch.device(cfg.device)
    print("🔥 Baseline 1: Transfer Learning (Warm-Start)")
    print("=" * 80)
    print("Strategy: Train on UNION of all training data (no task structure)")
    print("=" * 80)

    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}")

    # -----------------
    # CRITICAL: Load ALL training data (ignore task boundaries)
    # -----------------
    print("\n📦 Loading training data (Union of all tasks)...")
    
    train_ds = TrajectoryDataset(
        index_path=index_path,
        split="train",
        role="train_inner",
        check_shapes=True,
    )
    
    # Validation set (keep separate for monitoring)
    val_ds = TrajectoryDataset(
        index_path=index_path,
        split="val",
        role="val",
        check_shapes=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,  # CRITICAL: Shuffle destroys task structure
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(f"✅ Train trajectories: {len(train_ds)} (Mixed from all tasks)")
    print(f"✅ Val trajectories: {len(val_ds)}")

    # -----------------
    # Models (Same architecture as Meta-Learning)
    # -----------------
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    encoder = TrajEncoder(
        x_dim=x_dim,
        z_dim=z_dim,
        hidden_dim=cfg.latent.encoder_hidden_dim,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    sde = NeuralSDE(
        x_dim=x_dim,
        z_dim=z_dim,
        hidden_dim=cfg.latent.sde_hidden_dim,
    ).to(device)

    head = ForecastHead(
        x_dim=x_dim,
        z_dim=z_dim,
        hidden_dim=cfg.latent.head_hidden_dim,
    ).to(device)

    # Optimizer
    params = list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # RNG for Brownian noise
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.global_seed + 9999)  # Different seed from meta

    # Time grid
    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max_abs = cfg.stability.max_state_abs

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------
    # Training Loop (Identical to Meta-Learning)
    # -----------------
    print("\n🚀 Starting training...\n")

    for epoch in range(1, N_EPOCHS + 1):
        encoder.train()
        sde.train()
        head.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")

        for traj_batch, _ in pbar:
            traj_batch = traj_batch.to(device)
            B, T_total, d_ = traj_batch.shape

            # Encoder: observe first OBS_LEN steps
            obs = traj_batch[:, :OBS_LEN, :]
            z = encoder(obs)

            # SDE: simulate full trajectory
            x0 = traj_batch[:, 0, :]
            traj_pred = simulate_neural_sde_batch(
                sde=sde,
                x0=x0,
                z=z,
                T=T,
                n_steps=n_steps,
                x_max_abs=x_max_abs,
                generator=gen,
            )

            # Handle length mismatch
            if traj_pred.shape[1] != T_total:
                min_T = min(traj_pred.shape[1], T_total)
                traj_pred = traj_pred[:, :min_T, :]
                traj_true = traj_batch[:, :min_T, :]
            else:
                traj_true = traj_batch

            # Head: predict final state
            final_pred_sde = traj_pred[:, -1, :]
            head_pred = head(final_pred_sde, z)
            target_final = traj_true[:, -1, :]

            # Loss (same as meta-learning)
            loss_path = F.mse_loss(traj_pred, traj_true)
            loss_head = F.mse_loss(head_pred, target_final)
            loss = loss_path + 0.1 * loss_head

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        # -----------------
        # Validation
        # -----------------
        encoder.eval()
        sde.eval()
        head.eval()

        val_loss_total = 0.0
        with torch.no_grad():
            for traj_batch, _ in val_loader:
                traj_batch = traj_batch.to(device)
                B, T_total, _ = traj_batch.shape

                obs = traj_batch[:, :OBS_LEN, :]
                z = encoder(obs)

                x0 = traj_batch[:, 0, :]
                traj_pred = simulate_neural_sde_batch(
                    sde=sde,
                    x0=x0,
                    z=z,
                    T=T,
                    n_steps=n_steps,
                    x_max_abs=x_max_abs,
                    generator=gen,
                )

                if traj_pred.shape[1] != T_total:
                    min_T = min(traj_pred.shape[1], T_total)
                    traj_pred = traj_pred[:, :min_T, :]
                    traj_true = traj_batch[:, :min_T, :]
                else:
                    traj_true = traj_batch

                final_pred_sde = traj_pred[:, -1, :]
                head_pred = head(final_pred_sde, z)
                target_final = traj_true[:, -1, :]

                loss_path = F.mse_loss(traj_pred, traj_true)
                loss_head = F.mse_loss(head_pred, target_final)
                loss = loss_path + 0.1 * loss_head

                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"[Epoch {epoch}] Val Loss:   {avg_val_loss:.6f}")

        # -----------------
        # Checkpoint
        # -----------------
        if epoch % 10 == 0:
            ckpt_path = os.path.join("checkpoints", f"transfer_epoch_{epoch}.pt")
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "sde": sde.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"💾 Saved checkpoint to {ckpt_path}")

    print("\n" + "=" * 80)
    print("✅ Transfer Learning Training Complete!")
    print("=" * 80)
    print(f"Final checkpoint: checkpoints/transfer_epoch_{N_EPOCHS}.pt")
    print("\nNext step: Run baselines/adapt_transfer.py to test adaptation")


if __name__ == "__main__":
    train_transfer_baseline()
