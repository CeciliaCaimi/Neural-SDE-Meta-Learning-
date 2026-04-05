# baselines/train_MAML.py
# MAML-style meta-learning baseline on Neural SDE + Head
# No encoder, z = 0. First-order MAML with k inner steps.

import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

# --- CONFIG ---
INNER_STEPS = 5         # k inner-loop adaptation steps
INNER_LR = 1e-2
META_LR = 1e-3
META_BATCH_SIZE = 4     # tasks per meta-batch
EPOCHS = 20             # meta-epochs (you can adjust)
N_SHOTS = 2             # support trajectories per task

def get_task_indices(metadata):
    """Group dataset indices by theta_id."""
    groups = {}
    for idx, row in metadata.iterrows():
        tid = row["theta_id"]
        groups.setdefault(tid, []).append(idx)
    return groups

def sample_task_batch(ds, task_groups, device, n_shots):
    """Sample one task (theta_id) and split into support/query sets."""
    theta_id = random.choice(list(task_groups.keys()))
    idxs = task_groups[theta_id]

    # simplest: first N_SHOTS as support, remaining as query
    # assumes dataset ordering is already support/query-like per theta_id
    if len(idxs) < n_shots + 1:
        return None

    support_idxs = idxs[:n_shots]
    query_idxs = idxs[n_shots:]

    support = torch.stack([ds[i][0] for i in support_idxs]).to(device)
    query = torch.stack([ds[i][0] for i in query_idxs]).to(device)

    return support, query

def inner_adapt(sde, head, support, config, gen):
    """
    Perform k inner-loop gradient steps from the shared init (sde, head)
    on the support set, using final-step MSE and z=0.
    Returns adapted copies (sde_adapted, head_adapted).
    """
    device = support.device
    x_dim = config.basis.x_dim
    z_dim = config.latent.latent_dim

    # Copy parameters for inner-loop
    sde_fast = NeuralSDE(x_dim, z_dim, config.latent.sde_hidden_dim).to(device)
    head_fast = ForecastHead(x_dim, z_dim, config.latent.head_hidden_dim).to(device)
    sde_fast.load_state_dict(sde.state_dict())
    head_fast.load_state_dict(head.state_dict())

    optimizer = optim.SGD(
        list(sde_fast.parameters()) + list(head_fast.parameters()),
        lr=INNER_LR,
    )

    T = config.time_grid.T
    n_steps = config.time_grid.n_steps
    x_max = config.stability.max_state_abs

    x0 = support[:, 0, :]
    target_final = support[:, -1, :]
    z_zeros = torch.zeros(x0.size(0), z_dim, device=device)

    for _ in range(INNER_STEPS):
        optimizer.zero_grad()

        traj = simulate_neural_sde_batch(
            sde_fast, x0, z_zeros, T, n_steps, x_max, gen
        )
        pred_final = head_fast(traj[:, -1, :], z_zeros)

        loss_inner = F.mse_loss(pred_final, target_final)
        loss_inner.backward()
        optimizer.step()

    return sde_fast, head_fast

def main():
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.global_seed if hasattr(cfg, "global_seed") else 12345)
    random.seed(cfg.global_seed if hasattr(cfg, "global_seed") else 12345)

    print("🔥 MAML-style Baseline: Neural SDE + Head, z=0 (no encoder)")
    print(f"Inner steps: {INNER_STEPS}, inner_lr: {INNER_LR}, meta_lr: {META_LR}")
    print(f"Tasks per meta-batch: {META_BATCH_SIZE}, epochs: {EPOCHS}\n")

    # 1. Load meta-train data (use 'train' regime, 'train_inner' split)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    train_ds = TrajectoryDataset(index_path, "train", "train_inner", check_shapes=True)
    task_groups = get_task_indices(train_ds.metadata)

    # 2. Initialize model
    x_dim = cfg.basis.x_dim
    z_dim = cfg.latent.latent_dim

    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)

    meta_optimizer = optim.Adam(
        list(sde.parameters()) + list(head.parameters()),
        lr=META_LR,
    )

    T = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.global_seed if hasattr(cfg, "global_seed") else 12345)

    # 3. Meta-training loop (first-order MAML)
    for epoch in range(1, EPOCHS + 1):
        meta_loss_sum = 0.0
        num_meta_batches = 0

        # we just loop over a fixed number of meta-batches per epoch
        for _ in tqdm(range(100), desc=f"Meta-epoch {epoch}"):
            meta_optimizer.zero_grad()
            batch_meta_loss = 0.0
            tasks_used = 0

            for _ in range(META_BATCH_SIZE):
                sample = sample_task_batch(train_ds, task_groups, device, N_SHOTS)
                if sample is None:
                    continue
                support, query = sample
                tasks_used += 1

                # Inner adaptation
                sde_fast, head_fast = inner_adapt(sde, head, support, cfg, gen)

                # Compute query loss with adapted params
                x0_q = query[:, 0, :]
                target_final_q = query[:, -1, :]
                z_zeros_q = torch.zeros(x0_q.size(0), z_dim, device=device)

                traj_q = simulate_neural_sde_batch(
                    sde_fast, x0_q, z_zeros_q, T, n_steps, x_max, gen
                )
                pred_final_q = head_fast(traj_q[:, -1, :], z_zeros_q)

                loss_query = F.mse_loss(pred_final_q, target_final_q)
                # First-order MAML: treat sde_fast/head_fast as detached from second-order terms
                # Accumulate meta-loss as sum/mean over tasks
                batch_meta_loss += loss_query

            if tasks_used == 0:
                continue

            batch_meta_loss = batch_meta_loss / tasks_used
            batch_meta_loss.backward()
            meta_optimizer.step()

            meta_loss_sum += batch_meta_loss.item()
            num_meta_batches += 1

        if num_meta_batches > 0:
            print(f"Epoch {epoch} | Meta-loss: {meta_loss_sum / num_meta_batches:.4f}")

    # 4. Save meta-initialization
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "sde": sde.state_dict(),
            "head": head.state_dict(),
        },
        "checkpoints/maml_nsde_init.pt",
    )
    print("\n✅ Saved MAML NSDE initialization to checkpoints/maml_nsde_init.pt")

if __name__ == "__main__":
    main()
