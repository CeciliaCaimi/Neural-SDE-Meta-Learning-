# training/train_compositional.py
#
# Item 6, phase 3/4: train_base and train_adapt stages for the compositional-
# generalisation experiment's four arms (adaptation/compositional_arms.py).
#
# Deliberately a NEW, self-contained module rather than adding task-id-list
# parameters to training/train_meta.py / training/train_controller.py /
# training/train_diff_adaptation.py. Those three scripts' driver bodies are
# tightly coupled to the legacy regime-based data loading (TrajectoryDataset
# by split name, meta_params keyed by "test<regime>") and are relied upon
# unchanged by items 1-4's already-queued real-GPU comparisons; threading an
# arm/task-id-list branch through them would risk changing their behaviour
# for an unrelated experiment. Instead, this module reuses their REUSABLE
# inner building blocks directly (simulate_neural_sde_batch,
# euler_transition_nll, adapt_model, compute_residual, k_step_adapt,
# get_rates, via adaptation/compositional_arms.py) and supplies its own
# outer loop driven by an explicit task-id list (item 6 phase 1d) and
# dataloaders.factorial_dataset.FactorialTrajectoryDataset -- see
# COMPOSITIONAL_EXPERIMENT.md for this decision.

import os
import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from adaptation.compositional_arms import build_models, differentiable_adapt_factorial, encode
from adaptation.gated_finetuning_regularized import N_SHOTS, get_task_data
from dataloaders.trajectory_datasets import apply_scaler_to_trajectories, fit_scaler_on_trajectories
from training.losses import euler_transition_nll
from training.train_meta import simulate_neural_sde_batch

LEARNING_RATE = 1e-3
LAMBDA_HEAD = 0.1
LAMBDA_ROLLOUT = 1.0
LAMBDA_ENDPOINT = 0.1

LOSS_VARIANTS = (
    "trajectory_mse",
    "transition_nll",
    "transition_nll_rollout",
    "transition_nll_rollout_endpoint",
)
DEFAULT_LOSS_VARIANT = "trajectory_mse"


def train_base(
    arm: str,
    train_ds,
    x_dim: int,
    z_dim: int,
    cfg,
    loss_variant: str = DEFAULT_LOSS_VARIANT,
    n_epochs: int = 50,
    batch_size: int = 64,
    obs_len_range: Sequence[int] = (20, 50),
    checkpoint_path: str = "checkpoints/compositional_base.pt",
    max_steps_per_epoch: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
    return_losses: bool = False,
) -> Optional[List[float]]:
    """Train the base encoder/sde/head for one arm on an explicit training
    dataset (a FactorialTrajectoryDataset built from the compositional
    split's `train` task ids and role="train_inner").

    Mirrors training/train_meta.py's train_meta_loop training step exactly
    (same loss-variant selection, same simulate_neural_sde_batch call, same
    gradient-clipping Adam step), generalised over `arm` via
    adaptation.compositional_arms.encode() for how (z_f, z_g) are produced
    (tied for single_z, independent for the three factorised arms).
    """
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"Unknown loss_variant={loss_variant!r}, expected one of {LOSS_VARIANTS}")

    device = torch.device(device or cfg.device)
    print(f"Training base model for arm={arm!r} on {device} (loss_variant={loss_variant})")

    print("Fitting source scaler on compositional training split...")
    source_scaler = train_ds.fit_scaler()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder, sde, head, _ = build_models(
        arm, x_dim, z_dim, cfg.latent.encoder_hidden_dim, cfg.latent.sde_hidden_dim,
        cfg.latent.head_hidden_dim, device,
    )
    params = list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    T, n_steps, dt = cfg.time_grid.T, cfg.time_grid.n_steps, cfg.time_grid.dt
    x_max_abs = cfg.stability.max_state_abs

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    step_losses: List[float] = []
    for epoch in range(1, n_epochs + 1):
        encoder.train(); sde.train(); head.train()
        for step_idx, (traj_batch, _) in enumerate(train_loader):
            if max_steps_per_epoch is not None and step_idx >= max_steps_per_epoch:
                break
            traj_batch = traj_batch.to(device)
            B, T_total, d_ = traj_batch.shape

            current_obs_len = torch.randint(
                low=obs_len_range[0], high=obs_len_range[1] + 1, size=(1,)
            ).item()
            obs = traj_batch[:, :current_obs_len, :]
            z_f, z_g = encode(arm, encoder, obs)

            x0 = traj_batch[:, 0, :]
            traj_pred = simulate_neural_sde_batch(sde, x0, z_f, z_g, T, n_steps, x_max_abs, gen)

            if traj_pred.shape[1] != T_total:
                min_T = min(traj_pred.shape[1], T_total)
                traj_pred = traj_pred[:, :min_T, :]
                traj_true = traj_batch[:, :min_T, :]
            else:
                traj_true = traj_batch

            final_pred_sde = traj_pred[:, -1, :]
            head_pred = head(final_pred_sde, z_f, z_g)
            target_final = traj_true[:, -1, :]
            loss_head = F.mse_loss(head_pred, target_final)

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
                    loss_endpoint = F.mse_loss(traj_pred[:, -1, :], traj_true[:, -1, :])
                    loss_sde = loss_nll + LAMBDA_ROLLOUT * loss_rollout + LAMBDA_ENDPOINT * loss_endpoint

            loss = loss_sde + LAMBDA_HEAD * loss_head

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if return_losses:
                step_losses.append(loss.item())

    torch.save({
        "arm": arm,
        "z_dim": z_dim,
        "loss_variant": loss_variant,
        "encoder": encoder.state_dict(),
        "sde": sde.state_dict(),
        "head": head.state_dict(),
        "source_scaler": source_scaler,
    }, checkpoint_path)
    print(f"Saved base checkpoint: {checkpoint_path}")

    return step_losses if return_losses else None


def load_base_checkpoint(arm: str, checkpoint_path: str, x_dim: int, cfg, device):
    """Load a train_base() checkpoint, frozen (eval mode, requires_grad=False)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("arm", arm) != arm:
        raise ValueError(f"Checkpoint {checkpoint_path} was trained for arm={ckpt.get('arm')!r}, expected {arm!r}")
    z_dim = ckpt["z_dim"]
    encoder, sde, head, _ = build_models(
        arm, x_dim, z_dim, cfg.latent.encoder_hidden_dim, cfg.latent.sde_hidden_dim,
        cfg.latent.head_hidden_dim, device,
    )
    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])
    encoder.eval(); sde.eval(); head.eval()
    for p in list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters()):
        p.requires_grad = False
    return encoder, sde, head, z_dim, ckpt.get("source_scaler")


def train_adapt(
    arm: str,
    base_checkpoint_path: str,
    x_dim: int,
    theta_by_id: Dict[str, object],
    ds_supp,
    ds_query,
    train_task_ids: Sequence[str],
    cfg,
    n_meta_steps: int = 200,
    tasks_per_step: int = 4,
    lr: float = 1e-3,
    steps_available: int = 50,
    K: int = 8,
    rate_hidden_dim: int = 32,
    gate_hidden_dim: int = 32,
    out_path: Optional[str] = None,
    seed: int = 0,
    device: Optional[str] = None,
    return_diagnostics: bool = False,
) -> Optional[dict]:
    """Meta-train the adaptation-policy modules (rate_controller, and for
    factorised_selective also gate_controller) for arms 3/4.

    For single_z/factorised (fixed-rate adaptation, no meta-parameters)
    this is a no-op returning None -- consistent with
    training/train_diff_adaptation.py's adaptation_mode="fixed" having zero
    meta-parameters by construction.
    """
    if arm not in ("factorised_learned_rates", "factorised_selective"):
        return None

    device = torch.device(device or cfg.device)
    encoder, sde, head, z_dim, source_scaler = load_base_checkpoint(arm, base_checkpoint_path, x_dim, cfg, device)

    _, _, _, extra = build_models(
        arm, x_dim, z_dim, cfg.latent.encoder_hidden_dim, cfg.latent.sde_hidden_dim,
        cfg.latent.head_hidden_dim, device, rate_hidden_dim, gate_hidden_dim,
    )
    rate_controller = extra["rate_controller"]
    gate_controller = extra.get("gate_controller")
    adaptation_mode = "conditioned" if arm == "factorised_learned_rates" else "conditioned_plus_gate"

    meta_trainable = list(rate_controller.parameters())
    if gate_controller is not None:
        meta_trainable += list(gate_controller.parameters())
    optimizer = optim.Adam(meta_trainable, lr=lr)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    task_rng = random.Random(seed)

    T_full, n_steps, x_max = cfg.time_grid.T, cfg.time_grid.n_steps, cfg.stability.max_state_abs

    train_task_ids = list(train_task_ids)
    losses: List[float] = []
    grad_norms: List[float] = []

    for _ in range(n_meta_steps):
        batch_ids = [task_rng.choice(train_task_ids) for _ in range(tasks_per_step)]
        optimizer.zero_grad()
        task_losses = []

        for tid in batch_ids:
            theta = theta_by_id[tid]
            support_full = get_task_data(ds_supp, tid, device)[:N_SHOTS]
            query = get_task_data(ds_query, tid, device)

            target_scaler = fit_scaler_on_trajectories(support_full) if source_scaler is not None else None
            support = support_full[:, :steps_available]
            query_in = apply_scaler_to_trajectories(query, target_scaler) if target_scaler is not None else query

            zf_final, zg_final, _, _, _, _ = differentiable_adapt_factorial(
                encoder, sde, head, support, gen, cfg, theta, adaptation_mode,
                target_scaler=target_scaler, rate_controller=rate_controller,
                gate_controller=gate_controller, K=K,
            )

            B_q = query_in.shape[0]
            zf_q = zf_final.expand(B_q, -1)
            zg_q = zg_final.expand(B_q, -1)
            traj_pred = simulate_neural_sde_batch(sde, query_in[:, 0], zf_q, zg_q, T_full, n_steps, x_max, gen)
            valid_len = min(traj_pred.shape[1], query_in.shape[1])
            query_loss = F.mse_loss(traj_pred[:, :valid_len], query_in[:, :valid_len])
            task_losses.append(query_loss)

        if not task_losses:
            continue

        step_loss = torch.stack(task_losses).mean()
        step_loss.backward()
        grad_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in meta_trainable if p.grad is not None)
        ).item()
        grad_norms.append(grad_norm)
        optimizer.step()
        losses.append(step_loss.item())

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        state = {
            "arm": arm, "adaptation_mode": adaptation_mode, "z_dim": z_dim, "K": K,
            "rate_controller": rate_controller.state_dict(), "rate_hidden_dim": rate_hidden_dim,
        }
        if gate_controller is not None:
            state["gate_controller"] = gate_controller.state_dict()
            state["gate_hidden_dim"] = gate_hidden_dim
        torch.save(state, out_path)
        print(f"Saved adapt checkpoint: {out_path}")

    if return_diagnostics:
        return {"losses": losses, "grad_norms": grad_norms}
    return None


def load_adapt_extra(arm: str, adapt_checkpoint_path: str, z_dim: int, device):
    """Load a train_adapt() checkpoint into fresh rate_controller/gate_controller
    modules, frozen for evaluation."""
    from adaptation.compositional_arms import build_models as _build

    ckpt = torch.load(adapt_checkpoint_path, map_location=device, weights_only=False)
    rate_hidden_dim = ckpt.get("rate_hidden_dim", 32)
    gate_hidden_dim = ckpt.get("gate_hidden_dim", 32)
    _, _, _, extra = _build(arm, 1, z_dim, 1, 1, 1, device, rate_hidden_dim, gate_hidden_dim)
    extra["rate_controller"].load_state_dict(ckpt["rate_controller"])
    extra["rate_controller"].eval()
    if "gate_controller" in ckpt:
        extra["gate_controller"].load_state_dict(ckpt["gate_controller"])
        extra["gate_controller"].eval()
    return extra
