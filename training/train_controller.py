# training/train_controller.py
#
# Item 3, step 3: meta-train the LearnedGateController from
# adaptation/gate_controller.py.
#
# This is deliberately separate from training/train_meta.py (which trains
# encoder/sde/head) and from the per-task Adam fine-tuning in
# adaptation/gated_finetuning_regularized.py's adapt_model(). Those two are
# reused here as fixed, frozen building blocks:
#
#   1. encoder/sde/head are loaded from an existing checkpoint and frozen
#      (requires_grad=False, eval() mode) for the entire run -- this learns
#      an adaptation *policy* on top of an already-trained base model, not
#      a retraining of that model.
#   2. For each task, adapt_model() runs its usual ADAPT_STEPS of Adam
#      fine-tuning on a copied head + (z_f, z_g) to get the full-adaptation
#      deltas Delta z_f = zf_opt - zf_init, Delta z_g = zg_opt - zg_init.
#      adapt_model() already .detach()'s zf_opt/zg_opt before returning
#      them, so these deltas carry no gradient history into the controller
#      by construction.
#   3. The controller maps (zf_init, zg_init, drift/diffusion residual,
#      grad_norm_zf/zg) -> (a_f, a_g), which are used to blend
#      z_f' = zf_init + a_f * Delta z_f (same for z_g'). Only a_f/a_g
#      carry gradient into this expression, so backprop through the query
#      rollout loss flows into the controller's own parameters only --
#      encoder/sde/head are never updated here.
#
# --- Design choice: NOT differentiating through the inner Adam loop -----
# A "fuller" version of this idea would unroll (or implicit-diff through)
# adapt_model()'s ADAPT_STEPS of Adam updates, so the controller could also
# learn to shape *how* z_f/z_g adapt, not just how much of the already-
# computed delta to keep. We are deliberately NOT doing that here:
#   - The item 3 spec frames Delta z_f/Delta z_g as "already-computed" and
#     explicitly says the controller "does not need to backpropagate
#     through the inner Adam fine-tuning loop itself" -- treat them as
#     fixed/detached.
#   - Unrolling ADAPT_STEPS=50 Adam steps for every meta-step would be far
#     more expensive (50x the backward-pass depth per task) and Adam's
#     moment estimates make a naive unroll numerically delicate; doing this
#     properly wants either full backprop-through-unrolled-Adam or an
#     implicit-function-theorem treatment, which is a meaningfully bigger
#     undertaking than "add a selectable gate mode."
#   - The existing scalar gate already treats the post-adaptation latents
#     as given constants downstream (it never asks "how should adaptation
#     have gone differently") -- this controller keeps that same boundary,
#     just replacing the scalar reweighting with a learned, factor-specific
#     one.
# If a future iteration wants the fuller version, that's a distinct,
# larger design change, not an incremental extension of this script.
#
# --- Where the training tasks come from ---------------------------------
# The controller needs tasks with a genuine support/query split so it learns
# a policy that generalizes to held-out query data, the same way test-time
# adaptation itself is evaluated. Only "test<Regime>" splits have
# support/query roles (see dataloaders/trajectory_datasets.py); "train" only
# has train_inner/val_inner (used to train the base model itself, not to
# meta-train an adaptation policy). So this script trains on one such
# regime's (support, query) tasks, selected via --regime (default "none",
# i.e. testnone -- the in-distribution item-1 regime, so the controller
# first learns a policy on the least-shifted data before anyone evaluates
# it OOD). Generate that regime's thetas/trajectories first if you haven't:
#   python -m data_gen.generate_meta_params --regimes factorised
#   python -m data_gen.generate_trajectories
#
# --- Running the real thing ----------------------------------------------
# This module's smoke test (smoke_test_controller.py) only confirms the
# training loop runs end-to-end with finite, non-degenerate gradients at
# toy scale -- it is NOT a real training run. To actually meta-train the
# controller:
#   python -m training.train_controller \
#       --checkpoint checkpoints/meta_epoch_50.pt \
#       --regime none \
#       --n-meta-steps 2000 --tasks-per-step 8 \
#       --controller-out checkpoints/controller.pt
#
# Then compare it against the existing scalar gate across the five item-1
# regimes (none/drift_only/diffusion_only/combined/ood_1..5), using the
# SAME base checkpoint for both so only the gate differs:
#   python -m adaptation.gated_finetuning_regularized --regimes factorised \
#       --checkpoint checkpoints/meta_epoch_50.pt --gate-mode scalar
#   python -m adaptation.gated_finetuning_regularized --regimes factorised \
#       --checkpoint checkpoints/meta_epoch_50.pt --gate-mode learned \
#       --controller-checkpoint checkpoints/controller.pt
# This writes two separate CSVs (results/gated_regularized_final.csv and
# results/gated_controller_final.csv). Compare, per regime and steps_available:
#   - query-loss proxy: mse_rollout / rmse_rollout (what the controller is
#     actually trained to minimize)
#   - drift_error / diffusion_error (item 1's mechanism-identification
#     metrics -- the point of having per-factor (a_f, a_g) instead of one
#     shared g is to see whether the controller can, e.g., trust an
#     adapted z_f while distrusting an adapted z_g, or vice versa; if
#     drift_error/diffusion_error under "learned" don't improve over
#     "scalar" on any regime, the extra complexity isn't earning its keep)
#   - nll (uncertainty calibration)
# Only once this comparison shows "learned" is at least as good as
# "scalar" (ideally better on the regimes it's meant to help with) does it
# make sense to consider removing the scalar gate -- not before.

import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import (
    TrajectoryDataset,
    fit_scaler_on_trajectories,
    apply_scaler_to_trajectories,
)
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch
from adaptation.gated_finetuning_regularized import (
    adapt_model, compute_mechanism_error, get_task_data, N_SHOTS,
)
from adaptation.gate_controller import LearnedGateController, build_controller_input

DEFAULT_LR = 1e-3
DEFAULT_N_META_STEPS = 200
DEFAULT_TASKS_PER_STEP = 4
DEFAULT_STEPS_AVAILABLE = 50  # support-set length seen during controller training
DEFAULT_HIDDEN_DIM = 32


def controller_train_loop(
    checkpoint_path: str,
    regime: str = "none",
    n_meta_steps: int = DEFAULT_N_META_STEPS,
    tasks_per_step: int = DEFAULT_TASKS_PER_STEP,
    lr: float = DEFAULT_LR,
    steps_available: int = DEFAULT_STEPS_AVAILABLE,
    controller_hidden_dim: int = DEFAULT_HIDDEN_DIM,
    controller_out: str = "checkpoints/controller.pt",
    device: Optional[str] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> Optional[Dict[str, List[float]]]:
    """Meta-train a LearnedGateController on top of a frozen base checkpoint.

    Args:
        checkpoint_path: base model checkpoint (encoder/sde/head +
            source_scaler), produced by training/train_meta.py. Frozen for
            the entire run.
        regime: which "test<regime>" split's (support, query) tasks to
            train on (see module docstring for why test regimes, not train).
        n_meta_steps: number of controller optimizer steps.
        tasks_per_step: number of tasks averaged into each meta-step's loss
            (a task "batch", analogous to BATCH_SIZE in train_meta.py, but
            here each task requires its own adapt_model() call so this is a
            Python-level loop, not a single vectorized forward pass).
        steps_available: support-set length (in time-steps) used for
            adaptation during controller training, analogous to one entry
            of STEPS_SWEEP in gated_finetuning_regularized.py.
        return_diagnostics: if True, return per-meta-step query loss,
            (a_f, a_g) seen across tasks, and the controller's gradient
            norm after each backward() -- used by smoke_test_controller.py
            to confirm gradients actually reach the controller and that
            a_f/a_g are not degenerate. If False, returns None.
    """
    device = torch.device(device or cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])

    # Frozen throughout: this trains an adaptation policy on top of an
    # already-trained base model, not the base model itself.
    encoder.eval(); sde.eval(); head.eval()
    for p in list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters()):
        p.requires_grad = False

    source_scaler = ckpt.get("source_scaler", None)

    meta_params = torch.load(cfg.paths.meta_params_path, map_location="cpu", weights_only=False)
    regime_key = f"test{regime}"
    if regime_key not in meta_params:
        raise ValueError(
            f"No thetas for regime={regime_key!r} in {cfg.paths.meta_params_path}. "
            f"Generate it first, e.g. python -m data_gen.generate_meta_params --regimes factorised "
            f"followed by python -m data_gen.generate_trajectories."
        )
    theta_by_id = {t.id: t.to(device) for t in meta_params[regime_key]}

    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    ds_supp = TrajectoryDataset(index_path, regime_key, "support")
    ds_query = TrajectoryDataset(index_path, regime_key, "query")
    task_ids = list(ds_supp.metadata["theta_id"].unique())
    if len(task_ids) == 0:
        raise RuntimeError(f"No tasks found for regime={regime_key!r}.")

    controller = LearnedGateController(z_dim=z_dim, hidden_dim=controller_hidden_dim).to(device)
    optimizer = optim.Adam(controller.parameters(), lr=lr)

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    task_rng = torch.Generator().manual_seed(seed)

    T_full, n_steps, x_max = cfg.time_grid.T, cfg.time_grid.n_steps, cfg.stability.max_state_abs

    losses: List[float] = []
    a_f_history: List[float] = []
    a_g_history: List[float] = []
    controller_grad_norms: List[float] = []

    for meta_step in tqdm(range(n_meta_steps), desc="controller meta-step"):
        idx = torch.randint(len(task_ids), (tasks_per_step,), generator=task_rng)
        batch_task_ids = [task_ids[i] for i in idx.tolist()]

        optimizer.zero_grad()
        task_losses = []

        for theta_id in batch_task_ids:
            n_supp = int((ds_supp.metadata["theta_id"] == theta_id).sum())
            n_query = int((ds_query.metadata["theta_id"] == theta_id).sum())
            if n_supp == 0 or n_query == 0:
                continue
            theta = theta_by_id[theta_id]

            support_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query = get_task_data(ds_query, theta_id, device)

            target_scaler = (
                fit_scaler_on_trajectories(support_full) if source_scaler is not None else None
            )
            support = support_full[:, :steps_available]

            if target_scaler is not None:
                support_in = apply_scaler_to_trajectories(support, target_scaler)
                query_in = apply_scaler_to_trajectories(query, target_scaler)
            else:
                support_in = support
                query_in = query

            with torch.no_grad():
                enc_len = min(support_in.shape[1], 50)
                zf_all, zg_all = encoder(support_in[:, :enc_len])
                zf_init = zf_all.mean(dim=0, keepdim=True)
                zg_init = zg_all.mean(dim=0, keepdim=True)

            # Already-computed full-adaptation step, reused as-is. adapt_model
            # returns zf_opt/zg_opt already .detach()'d, and grad_norm_zf/zg
            # as plain floats -- nothing here carries gradient history back
            # into this inner loop (see the design-choice note above).
            _, zf_opt, zg_opt, _, grad_norm_zf, grad_norm_zg = adapt_model(
                sde, head, zf_init, zg_init, support_in, gen, cfg
            )

            drift_residual, diffusion_residual = compute_mechanism_error(
                sde, theta, zf_init, zg_init, support_in, support, target_scaler
            )

            ctrl_input = build_controller_input(
                zf_init, zg_init, drift_residual, diffusion_residual,
                grad_norm_zf, grad_norm_zg,
            )
            a = controller(ctrl_input)  # (1, 2) -- requires_grad via controller params
            a_f, a_g = a[0, 0], a[0, 1]
            a_f_history.append(a_f.item())
            a_g_history.append(a_g.item())

            zf_pred = zf_init + a_f * (zf_opt - zf_init)
            zg_pred = zg_init + a_g * (zg_opt - zg_init)

            B_q = query_in.shape[0]
            zf_query = zf_pred.expand(B_q, -1)
            zg_query = zg_pred.expand(B_q, -1)
            traj_pred = simulate_neural_sde_batch(
                sde, query_in[:, 0], zf_query, zg_query, T_full, n_steps, x_max, gen
            )
            valid_len = min(traj_pred.shape[1], query_in.shape[1])
            query_loss = F.mse_loss(traj_pred[:, :valid_len], query_in[:, :valid_len])
            task_losses.append(query_loss)

        if not task_losses:
            continue

        step_loss = torch.stack(task_losses).mean()
        step_loss.backward()

        grad_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in controller.parameters() if p.grad is not None)
        ).item()
        controller_grad_norms.append(grad_norm)

        optimizer.step()
        losses.append(step_loss.item())

    os.makedirs(os.path.dirname(controller_out) or ".", exist_ok=True)
    torch.save(
        {
            "controller": controller.state_dict(),
            "z_dim": z_dim,
            "hidden_dim": controller_hidden_dim,
            "regime": regime,
        },
        controller_out,
    )
    print(f"Saved controller checkpoint to {controller_out}")

    if return_diagnostics:
        return {
            "losses": losses,
            "a_f_history": a_f_history,
            "a_g_history": a_g_history,
            "controller_grad_norms": controller_grad_norms,
        }
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-train the learned drift/diffusion adaptation controller (item 3)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint path.")
    parser.add_argument(
        "--regime", type=str, default="none",
        help="Which 'test<regime>' split to train on (default: none, the item-1 in-distribution regime).",
    )
    parser.add_argument("--n-meta-steps", type=int, default=DEFAULT_N_META_STEPS)
    parser.add_argument("--tasks-per-step", type=int, default=DEFAULT_TASKS_PER_STEP)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps-available", type=int, default=DEFAULT_STEPS_AVAILABLE)
    parser.add_argument("--controller-hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--controller-out", type=str, default="checkpoints/controller.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    controller_train_loop(
        checkpoint_path=args.checkpoint,
        regime=args.regime,
        n_meta_steps=args.n_meta_steps,
        tasks_per_step=args.tasks_per_step,
        lr=args.lr,
        steps_available=args.steps_available,
        controller_hidden_dim=args.controller_hidden_dim,
        controller_out=args.controller_out,
        seed=args.seed,
    )
