# training/train_meta.py
# Meta-Training Loop with Randomized Context (Robustness Fix)
#
# --- Full pipeline (item 1: factorised z_f/z_g model) ---
# 1. Generate theta parameters for train/val plus whichever test regimes
#    you want. The legacy combined-shift regimes (A, B, C) are the default;
#    pass --regimes factorised to also generate the item-1 regime set
#    (none, drift_only, diffusion_only, combined, ood_1..ood_5):
#       python -m data_gen.generate_meta_params --regimes factorised
# 2. Simulate trajectories and build data/index.csv for every regime that
#    was just generated (this script is regime-name agnostic, no flag
#    needed):
#       python -m data_gen.generate_trajectories
# 3. Train the factorised model (this script). No regime flag needed here —
#    training only touches the train/val split; checkpoints land in
#    checkpoints/meta_epoch_{10,20,...}.pt (gitignored):
#       python -m training.train_meta
# 4. Run test-time adaptation + evaluation, selecting which regimes to
#    score against with --regimes (default is the legacy testA/testB/testC;
#    pass --regimes factorised for the item-1 regime set):
#       python -m adaptation.gated_finetuning_regularized --regimes factorised
#
# Expect step 2 (trajectory simulation) and step 3 (training, 50 epochs
# over cfg.dataset_sizes.n_train_thetas thetas x ~80 trajectories each) to be
# the compute-heavy stages — run those on a GPU box. Steps 1 and 4 are cheap
# by comparison (theta sampling, and a handful of Adam steps per test task).
#
# --- Item 2: selectable training objective ---
# Step 3 (this script) selects the SDE training objective via
# --loss-variant (see LOSS_VARIANTS below); default is trajectory_mse,
# i.e. unchanged pre-item-2 behavior. The other three variants add the
# Euler-transition Gaussian NLL (training/losses.py), which scores drift
# and diffusion directly against every observed one-step transition instead
# of only the compounded rollout:
#       python -m training.train_meta --loss-variant trajectory_mse
#       python -m training.train_meta --loss-variant transition_nll
#       python -m training.train_meta --loss-variant transition_nll_rollout
#       python -m training.train_meta --loss-variant transition_nll_rollout_endpoint
#
# --- Running the real 4-way comparison (do this once, on GPU) ---
# This has NOT been run yet — the smoke test (smoke_test_loss_variants.py)
# only confirms each variant runs end-to-end at toy scale, not which one is
# best. To actually compare them:
#   1. Generate data once — steps 1-2 above don't depend on loss_variant.
#   2. Train one checkpoint per variant, pointing each at its own
#      --checkpoint-dir so they don't overwrite each other, e.g.:
#        python -m training.train_meta --loss-variant trajectory_mse                  --checkpoint-dir checkpoints_mse
#        python -m training.train_meta --loss-variant transition_nll                  --checkpoint-dir checkpoints_nll
#        python -m training.train_meta --loss-variant transition_nll_rollout          --checkpoint-dir checkpoints_nll_rollout
#        python -m training.train_meta --loss-variant transition_nll_rollout_endpoint --checkpoint-dir checkpoints_nll_rollout_endpoint
#   3. Evaluate each checkpoint with the existing item-1 metrics:
#        python -m adaptation.gated_finetuning_regularized --regimes factorised --checkpoint checkpoints_<variant>/meta_epoch_50.pt
#   4. Compare variants using held-out rollout error (mse_rollout/
#      rmse_rollout) AND drift_error/diffusion_error per regime (these are
#      the metrics that actually test whether the NLL objective identifies
#      the mechanisms better than MSE alone — don't judge on rollout error
#      alone, since that's what trajectory_mse is already directly
#      optimizing). Also sanity-check that "nll" in the eval CSV didn't
#      blow up for the NLL-trained variants on any regime.

import os
from typing import List, Optional, Tuple

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
from training.losses import euler_transition_nll

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 50
OBS_LEN = 50       # Max context training sees
LAMBDA_HEAD = 0.1  # Weight for forecasting loss

# -----------------------------
# Item 2: SDE training-objective variants
# -----------------------------
# "trajectory_mse" is the default / baseline (identical to pre-item-2
# behavior). The other three add the Euler-transition Gaussian NLL
# (training/losses.py). Which one is actually best is decided by a real
# GPU comparison (see module docstring above) — not here, and the default
# stays trajectory_mse until that comparison picks a winner.
LOSS_VARIANTS = (
    "trajectory_mse",                    # (a) existing rollout MSE only
    "transition_nll",                    # (b) Euler-transition Gaussian NLL only
    "transition_nll_rollout",            # (c) transition NLL + rollout MSE
    "transition_nll_rollout_endpoint",   # (d) transition NLL + rollout + endpoint MSE
)
DEFAULT_LOSS_VARIANT = "trajectory_mse"

# Weight on the rollout MSE term for variants (c)/(d). The NLL is in
# per-transition nats and the rollout loss is in normalized-state MSE units,
# with no real training run yet to calibrate a ratio between them — an
# unweighted sum (1.0) is the least assumption-laden default; retune once
# the real GPU comparison is in.
LAMBDA_ROLLOUT = 1.0

# Weight on the endpoint auxiliary loss for variant (d). Mirrors LAMBDA_HEAD
# above (also a single-final-state MSE term competing against full-path-scale
# terms) rather than introducing an unrelated new default.
LAMBDA_ENDPOINT = 0.1

# -----------------------------
# Simulator Function
# -----------------------------
def simulate_neural_sde_batch(
    sde: NeuralSDE,
    x0: torch.Tensor,         # (batch, d)
    z_f: torch.Tensor,        # (batch, zf_dim) — drift latent
    z_g: torch.Tensor,        # (batch, zg_dim) — diffusion latent
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

        # Drift and diffusion, each conditioned on its own latent
        b = sde.f(t, x, z_f)             # (batch, d)
        G = sde.g(t, x, z_g)             # (batch, d, d) diagonal
        
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
def train_meta_loop(
    loss_variant: str = DEFAULT_LOSS_VARIANT,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    obs_len_range: Tuple[int, int] = (20, OBS_LEN),
    index_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 10,
    max_steps_per_epoch: Optional[int] = None,
    return_losses: bool = False,
) -> Optional[List[float]]:
    """
    Args:
        loss_variant: one of LOSS_VARIANTS (item 2) — selects the SDE
            training objective. Default "trajectory_mse" reproduces the
            pre-item-2 behavior exactly.
        n_epochs, batch_size, obs_len_range, index_path, checkpoint_dir,
            checkpoint_every, max_steps_per_epoch: overridable so the same
            loop drives both the real training job (defaults) and a
            toy-scale smoke test (small overrides), without a second copy
            of the training logic. See smoke_test_loss_variants.py.
        return_losses: if True, return the list of per-step total losses
            instead of None (used by the smoke test to check finiteness).
    """
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss_variant={loss_variant!r}, expected one of {LOSS_VARIANTS}")

    device = torch.device(cfg.device)
    print(f"🚀 Starting meta-training on {device} (loss_variant={loss_variant})")

    if index_path is None:
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train trajectories: {len(train_ds)} | Val trajectories: {len(val_ds)}")

    # -----------------
    # Models
    # -----------------
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim
    
    # Ensure you increased encoder_hidden_dim in config/base_config.py!
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim, num_layers=2, dropout=0.1).to(device)
    sde = NeuralSDE(x_dim, z_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    params = list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.global_seed + 4242)

    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    dt = cfg.time_grid.dt
    x_max_abs = cfg.stability.max_state_abs

    os.makedirs(checkpoint_dir, exist_ok=True)

    step_losses: List[float] = []

    # -----------------
    # Training Loop
    # -----------------
    for epoch in range(1, n_epochs + 1):
        encoder.train()
        sde.train()
        head.train()

        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{n_epochs}")

        for step_idx, (traj_batch, _) in pbar:
            if max_steps_per_epoch is not None and step_idx >= max_steps_per_epoch:
                break

            traj_batch = traj_batch.to(device)
            B, T_total, d_ = traj_batch.shape

            # -------- A. Encoder: Randomized Context (Robustness) --------
            # FIX: Randomly sample length between obs_len_range.
            # This teaches the encoder to handle "starvation" (short inputs).
            current_obs_len = torch.randint(
                low=obs_len_range[0], high=obs_len_range[1] + 1, size=(1,)
            ).item()

            obs = traj_batch[:, :current_obs_len, :]
            z_f, z_g = encoder(obs)

            # -------- B. Neural SDE: Simulate Full Path --------
            x0 = traj_batch[:, 0, :]
            traj_pred = simulate_neural_sde_batch(
                sde, x0, z_f, z_g, T, n_steps, x_max_abs, gen
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
            head_pred = head(final_pred_sde, z_f, z_g)
            target_final = traj_true[:, -1, :]
            loss_head = F.mse_loss(head_pred, target_final)

            # -------- D. Loss: item-2 selectable SDE objective --------
            # loss_rollout (trajectory MSE) is always computed — it's both
            # variant (a) on its own and a term inside (c)/(d). The forecast
            # head's loss_head is unrelated to which SDE objective is chosen
            # (it trains ForecastHead, not drift_net/diff_net) so it's added
            # on top of every variant, same as pre-item-2.
            loss_rollout = F.mse_loss(traj_pred, traj_true)

            if loss_variant == "trajectory_mse":
                loss_sde = loss_rollout
            else:
                loss_nll = euler_transition_nll(sde, traj_true, z_f, z_g, dt)
                if loss_variant == "transition_nll":
                    loss_sde = loss_nll
                elif loss_variant == "transition_nll_rollout":
                    loss_sde = loss_nll + LAMBDA_ROLLOUT * loss_rollout
                else:  # transition_nll_rollout_endpoint
                    # Endpoint auxiliary loss: mirrors mse_final in
                    # adaptation/gated_finetuning_regularized.py
                    # (F.mse_loss(mean[:, -1], query_orig[:, -1])), just
                    # computed on the training rollout instead of an
                    # adapted eval rollout.
                    loss_endpoint = F.mse_loss(traj_pred[:, -1, :], traj_true[:, -1, :])
                    loss_sde = loss_nll + LAMBDA_ROLLOUT * loss_rollout + LAMBDA_ENDPOINT * loss_endpoint

            loss = loss_sde + LAMBDA_HEAD * loss_head

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            if return_losses:
                step_losses.append(loss.item())

        # -----------------
        # Validation & Save
        # -----------------
        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"meta_epoch_{epoch}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "sde": sde.state_dict(),
                "head": head.state_dict(),
                "cfg": cfg,
                "source_scaler": source_scaler,  # per-dim StandardScaler fitted on training split
                "loss_variant": loss_variant,
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    return step_losses if return_losses else None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meta-train the factorised (z_f, z_g) Neural SDE."
    )
    parser.add_argument(
        "--loss-variant",
        type=str,
        default=DEFAULT_LOSS_VARIANT,
        choices=LOSS_VARIANTS,
        help=(
            "SDE training objective (item 2). Default: trajectory_mse "
            "(unchanged pre-item-2 behavior). See the module docstring for "
            "what each variant means and how to run the real comparison."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Where to save checkpoints (default: checkpoints/). Use a distinct "
             "directory per variant when running the real comparison.",
    )
    args = parser.parse_args()

    train_meta_loop(loss_variant=args.loss_variant, checkpoint_dir=args.checkpoint_dir)
