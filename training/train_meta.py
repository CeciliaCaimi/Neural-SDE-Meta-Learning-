# training/train_meta.py
# Meta-Training Loop with Randomized Context (Robustness Fix)

import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 50
OBS_LEN = 50       # Max context training sees
LAMBDA_HEAD = 0.1  # Weight for forecasting loss

# -----------------------------
# Simulator Function
# -----------------------------
def simulate_neural_sde_batch(
    sde: NeuralSDE,
    x0: torch.Tensor,         # (batch, d)
    z: torch.Tensor,          # (batch, z_dim)
    T: float,
    n_steps: int,
    x_max_abs: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Euler–Maruyama simulation using the learned NeuralSDE.
    Returns: traj: (batch, n_steps + 1, d)
    """
    device = x0.device
    batch_size, d = x0.shape
    dt = T / n_steps
    sqrt_dt = dt ** 0.5
    
    traj = torch.zeros(batch_size, n_steps + 1, d, device=device)
    x = x0.clone()
    traj[:, 0, :] = x
    
    for k in range(n_steps):
        t = torch.tensor(k * dt, device=device)
        
        # Drift and diffusion
        b = sde.f(t, x, z)              # (batch, d)
        G = sde.g(t, x, z)              # (batch, d, d) diagonal
        
        # Brownian increment
        dW = torch.randn(batch_size, d, device=device, generator=generator) * sqrt_dt
        
        # Noise term: G @ dW
        noise = torch.bmm(G, dW.unsqueeze(-1)).squeeze(-1)
        
        # EM step
        x = x + b * dt + noise
        
        # Clamp state for numerical stability
        x = torch.clamp(x, -x_max_abs, x_max_abs)
        
        traj[:, k + 1, :] = x
        
    return traj


# -----------------------------
# Training Loop
# -----------------------------
def train_meta_loop():
    device = torch.device(cfg.device)
    print(f"🚀 Starting meta-training on {device}")
    
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.csv not found at {index_path}.")

    # -----------------
    # Datasets
    # -----------------
    train_ds = TrajectoryDataset(index_path, "train", "train_inner", check_shapes=True)

    # Two-scalar approach: fit the SOURCE scaler on training data only.
    # This scaler is saved with every checkpoint so evaluation scripts can
    # load it and maintain the correct coordinate frame.
    print("Fitting source scaler on training split (per-dimension StandardScaler)...")
    source_scaler = train_ds.fit_scaler()
    print(f"  mean range [{source_scaler.mean_.min():.4f}, {source_scaler.mean_.max():.4f}]  "
          f"std range [{source_scaler.scale_.min():.4f}, {source_scaler.scale_.max():.4f}]")

    val_ds = TrajectoryDataset(index_path, "val", "val", check_shapes=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train trajectories: {len(train_ds)} | Val trajectories: {len(val_ds)}")

    # -----------------
    # Models
    # -----------------
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    
    # Ensure you increased encoder_hidden_dim in config/base_config.py!
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim, num_layers=2, dropout=0.1).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    params = list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.global_seed + 4242)

    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max_abs = cfg.stability.max_state_abs

    os.makedirs("checkpoints", exist_ok=True)

    # -----------------
    # Training Loop
    # -----------------
    for epoch in range(1, N_EPOCHS + 1):
        encoder.train()
        sde.train()
        head.train()
        
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")

        for traj_batch, _ in pbar:
            traj_batch = traj_batch.to(device)
            B, T_total, d_ = traj_batch.shape

            # -------- A. Encoder: Randomized Context (Robustness) --------
            # FIX: Randomly sample length between 20 and 50.
            # This teaches the encoder to handle "starvation" (short inputs).
            current_obs_len = torch.randint(low=20, high=OBS_LEN + 1, size=(1,)).item()
            
            obs = traj_batch[:, :current_obs_len, :] 
            z = encoder(obs)

            # -------- B. Neural SDE: Simulate Full Path --------
            x0 = traj_batch[:, 0, :]
            traj_pred = simulate_neural_sde_batch(
                sde, x0, z, T, n_steps, x_max_abs, gen
            )

            # Align lengths if needed
            if traj_pred.shape[1] != T_total:
                min_T = min(traj_pred.shape[1], T_total)
                traj_pred = traj_pred[:, :min_T, :]
                traj_true = traj_batch[:, :min_T, :]
            else:
                traj_true = traj_batch

            # -------- C. Forecast Head --------
            final_pred_sde = traj_pred[:, -1, :]
            head_pred = head(final_pred_sde, z)
            target_final = traj_true[:, -1, :]

            # -------- D. Loss --------
            loss_path = F.mse_loss(traj_pred, traj_true)
            loss_head = F.mse_loss(head_pred, target_final)
            loss = loss_path + LAMBDA_HEAD * loss_head

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # -----------------
        # Validation & Save
        # -----------------
        if epoch % 10 == 0:
            ckpt_path = os.path.join("checkpoints", f"meta_epoch_{epoch}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "sde": sde.state_dict(),
                "head": head.state_dict(),
                "cfg": cfg,
                "source_scaler": source_scaler,  # per-dim StandardScaler fitted on training split
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train_meta_loop()