# training/train_diff_adaptation.py
#
# Item 4, meta-training loop: differentiate THROUGH the K-step adaptation
# loop from adaptation/differentiable_adaptation.py to meta-train
# component-specific adaptation rates (and, for "conditioned_plus_gate", a
# freshly-initialised (a_f, a_g) gate) via post-adaptation query loss.
#
# This is deliberately separate from training/train_controller.py (item 3),
# which meta-trains (a_f, a_g) against the existing 50-step Adam loop's
# DETACHED deltas. Here there is no detach anywhere in the inner loop: the
# query loss below backprops through K full trajectory simulations to reach
# the rate-conditioning meta-parameters. See
# adaptation/differentiable_adaptation.py's module docstring for the exact
# update rule, the choice of K, and how the four --adaptation-mode variants
# differ.
#
# encoder/sde/head are loaded from an existing checkpoint and kept FROZEN for
# the entire run in all four modes -- this meta-learns the adaptation
# process on top of an already-trained base model, not the base model
# itself (same boundary as train_controller.py).
#
# --- Running the real thing (GPU) ------------------------------------------
# This module's smoke test (smoke_test_diff_adaptation.py) only confirms, at
# toy scale, that gradients reach the meta-parameters of "global"/
# "conditioned"/"conditioned_plus_gate" and that "fixed" truly has none to
# update -- it is NOT a real training run or a real comparison. To actually
# meta-train and compare the four variants:
#
#   for mode in fixed global conditioned conditioned_plus_gate; do
#     python -m training.train_diff_adaptation \
#         --checkpoint checkpoints/meta_epoch_50.pt \
#         --regime none --adaptation-mode $mode \
#         --n-meta-steps 2000 --tasks-per-step 8 \
#         --out checkpoints/diff_adapt_${mode}.pt
#   done
#
# ("fixed" has no meta-parameters to save; it still runs the full loop and
# reports the query-loss trajectory so it can be used as the control curve.)
#
# Then compare all four, and against the existing scalar gate / item 3's
# controller, across the five item-1 regimes (none, drift_only,
# diffusion_only, combined, ood_1..5) on:
#   - post-adaptation query loss (what every "global"/"conditioned"/
#     "conditioned_plus_gate" variant is directly meta-trained to minimize;
#     "fixed" is the no-meta-learning control for this same quantity)
#   - item 1's mechanism-identification metrics: drift_error, diffusion_error
#     (adaptation.gated_finetuning_regularized.compute_mechanism_error),
#     plus rollout error and ||Delta z_f||, ||Delta z_g||
# Only once "conditioned" clearly beats "global" (task-conditioning earning
# its keep) and/or "conditioned_plus_gate" clearly beats "conditioned" (the
# added gate earning its keep) on these metrics -- and only once THOSE gains
# themselves plateau -- does it make sense to consider a recurrent/LSTM
# learned optimiser (an explicit non-goal of this module; see
# adaptation/differentiable_adaptation.py's module docstring).

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
from adaptation.gated_finetuning_regularized import get_task_data, N_SHOTS
from adaptation.gate_controller import LearnedGateController
from adaptation.differentiable_adaptation import (
    ADAPTATION_MODES,
    DEFAULT_K,
    GlobalRates,
    RateController,
    differentiable_adapt,
)

DEFAULT_LR = 1e-3
DEFAULT_N_META_STEPS = 200
DEFAULT_TASKS_PER_STEP = 4
DEFAULT_STEPS_AVAILABLE = 50
DEFAULT_RATE_HIDDEN_DIM = 32
DEFAULT_GATE_HIDDEN_DIM = 32


def diff_adaptation_train_loop(
    checkpoint_path: str,
    adaptation_mode: str,
    regime: str = "none",
    n_meta_steps: int = DEFAULT_N_META_STEPS,
    tasks_per_step: int = DEFAULT_TASKS_PER_STEP,
    lr: float = DEFAULT_LR,
    steps_available: int = DEFAULT_STEPS_AVAILABLE,
    K: int = DEFAULT_K,
    rate_hidden_dim: int = DEFAULT_RATE_HIDDEN_DIM,
    gate_hidden_dim: int = DEFAULT_GATE_HIDDEN_DIM,
    out_path: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> Optional[Dict[str, List[float]]]:
    """Meta-train item 4's differentiable K-step adaptation on top of a
    frozen base checkpoint, for one of the four --adaptation-mode variants.

    Args:
        adaptation_mode: one of ADAPTATION_MODES ("fixed", "global",
            "conditioned", "conditioned_plus_gate"). "fixed" has zero
            meta-parameters by construction -- this function still runs the
            full K-step loop and reports query loss (as a control curve),
            it just never constructs an optimizer.
        K: number of inner adaptation steps (see
            adaptation/differentiable_adaptation.py for why 8 by default).
        return_diagnostics: if True, return per-meta-step query loss and the
            meta-parameter gradient norm after each backward() -- used by
            smoke_test_diff_adaptation.py to confirm gradients actually
            reach the meta-parameters (and that "fixed" truly has none).
    """
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(f"Unknown adaptation_mode={adaptation_mode!r}, expected one of {ADAPTATION_MODES}")

    device = torch.device(device or cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    sde.load_state_dict(ckpt["sde"])
    head.load_state_dict(ckpt["head"])

    # Frozen throughout, in all four variants: this meta-learns the
    # adaptation process on top of an already-trained base model.
    encoder.eval(); sde.eval(); head.eval()
    for p in list(encoder.parameters()) + list(sde.parameters()) + list(head.parameters()):
        p.requires_grad = False

    source_scaler = ckpt.get("source_scaler", None)

    meta_params_data = torch.load(cfg.paths.meta_params_path, map_location="cpu", weights_only=False)
    regime_key = f"test{regime}"
    if regime_key not in meta_params_data:
        raise ValueError(
            f"No thetas for regime={regime_key!r} in {cfg.paths.meta_params_path}. "
            f"Generate it first, e.g. python -m data_gen.generate_meta_params --regimes factorised "
            f"followed by python -m data_gen.generate_trajectories."
        )
    theta_by_id = {t.id: t.to(device) for t in meta_params_data[regime_key]}

    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    ds_supp = TrajectoryDataset(index_path, regime_key, "support")
    ds_query = TrajectoryDataset(index_path, regime_key, "query")
    task_ids = list(ds_supp.metadata["theta_id"].unique())
    if len(task_ids) == 0:
        raise RuntimeError(f"No tasks found for regime={regime_key!r}.")

    # Construct only the meta-parameters this mode actually needs. "fixed"
    # constructs none of these -- meta_trainable_params stays empty, so no
    # optimizer is created below and nothing can be updated (the item-4
    # sanity check for variant (a): it is truly just a fixed-rate control).
    global_rates = rate_controller = gate_controller = None
    meta_trainable_params: List[torch.nn.Parameter] = []
    if adaptation_mode == "global":
        global_rates = GlobalRates().to(device)
        meta_trainable_params = list(global_rates.parameters())
    elif adaptation_mode == "conditioned":
        rate_controller = RateController(z_dim=z_dim, hidden_dim=rate_hidden_dim).to(device)
        meta_trainable_params = list(rate_controller.parameters())
    elif adaptation_mode == "conditioned_plus_gate":
        rate_controller = RateController(z_dim=z_dim, hidden_dim=rate_hidden_dim).to(device)
        # A FRESH LearnedGateController instance -- item 3's saved weights
        # were trained against a different, detached 50-step inner loop and
        # are not meaningful here; this one is trained from scratch, jointly
        # with rate_controller, inside this differentiable K-step loop.
        gate_controller = LearnedGateController(z_dim=z_dim, hidden_dim=gate_hidden_dim).to(device)
        meta_trainable_params = list(rate_controller.parameters()) + list(gate_controller.parameters())

    optimizer = optim.Adam(meta_trainable_params, lr=lr) if meta_trainable_params else None

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    task_rng = torch.Generator().manual_seed(seed)

    T_full, n_steps, x_max = cfg.time_grid.T, cfg.time_grid.n_steps, cfg.stability.max_state_abs

    losses: List[float] = []
    meta_grad_norms: List[float] = []

    for meta_step in tqdm(range(n_meta_steps), desc=f"diff-adapt[{adaptation_mode}] meta-step"):
        idx = torch.randint(len(task_ids), (tasks_per_step,), generator=task_rng)
        batch_task_ids = [task_ids[i] for i in idx.tolist()]

        if optimizer is not None:
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
            query_in = apply_scaler_to_trajectories(query, target_scaler) if target_scaler is not None else query

            zf_final, zg_final, _, _ = differentiable_adapt(
                encoder, sde, head, support, gen, cfg, theta, adaptation_mode,
                target_scaler=target_scaler, rate_controller=rate_controller,
                global_rates=global_rates, gate_controller=gate_controller, K=K,
            )

            B_q = query_in.shape[0]
            zf_query = zf_final.expand(B_q, -1)
            zg_query = zg_final.expand(B_q, -1)
            traj_pred = simulate_neural_sde_batch(
                sde, query_in[:, 0], zf_query, zg_query, T_full, n_steps, x_max, gen
            )
            valid_len = min(traj_pred.shape[1], query_in.shape[1])
            query_loss = F.mse_loss(traj_pred[:, :valid_len], query_in[:, :valid_len])
            task_losses.append(query_loss)

        if not task_losses:
            continue

        step_loss = torch.stack(task_losses).mean()

        if optimizer is not None:
            step_loss.backward()
            grad_norm = torch.sqrt(
                sum(p.grad.pow(2).sum() for p in meta_trainable_params if p.grad is not None)
            ).item()
            meta_grad_norms.append(grad_norm)
            optimizer.step()
        else:
            # "fixed": nothing to backprop into -- still confirm the query
            # loss is a valid finite scalar produced by the K-step loop.
            meta_grad_norms.append(0.0)

        losses.append(step_loss.item())

    if out_path and meta_trainable_params:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        state = {"adaptation_mode": adaptation_mode, "z_dim": z_dim, "K": K, "regime": regime}
        if global_rates is not None:
            state["global_rates"] = global_rates.state_dict()
        if rate_controller is not None:
            state["rate_controller"] = rate_controller.state_dict()
            state["rate_hidden_dim"] = rate_hidden_dim
        if gate_controller is not None:
            state["gate_controller"] = gate_controller.state_dict()
            state["gate_hidden_dim"] = gate_hidden_dim
        torch.save(state, out_path)
        print(f"Saved diff-adaptation checkpoint to {out_path}")
    elif out_path:
        print(f"adaptation_mode='fixed' has no meta-parameters to save; not writing {out_path}.")

    if return_diagnostics:
        return {
            "losses": losses,
            "meta_grad_norms": meta_grad_norms,
            "n_meta_params": sum(p.numel() for p in meta_trainable_params),
        }
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meta-train item 4's differentiable K-step adaptation (rates +/- gate)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint path.")
    parser.add_argument(
        "--adaptation-mode", type=str, required=True, choices=list(ADAPTATION_MODES),
        help="Which of the four item-4 variants to run/meta-train.",
    )
    parser.add_argument(
        "--regime", type=str, default="none",
        help="Which 'test<regime>' split to train on (default: none, the item-1 in-distribution regime).",
    )
    parser.add_argument("--n-meta-steps", type=int, default=DEFAULT_N_META_STEPS)
    parser.add_argument("--tasks-per-step", type=int, default=DEFAULT_TASKS_PER_STEP)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--steps-available", type=int, default=DEFAULT_STEPS_AVAILABLE)
    parser.add_argument("--k-steps", type=int, default=DEFAULT_K, help="Number of inner adaptation steps K.")
    parser.add_argument("--rate-hidden-dim", type=int, default=DEFAULT_RATE_HIDDEN_DIM)
    parser.add_argument("--gate-hidden-dim", type=int, default=DEFAULT_GATE_HIDDEN_DIM)
    parser.add_argument("--out", type=str, default=None, help="Where to save the meta-trained checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = args.out or f"checkpoints/diff_adapt_{args.adaptation_mode}.pt"

    diff_adaptation_train_loop(
        checkpoint_path=args.checkpoint,
        adaptation_mode=args.adaptation_mode,
        regime=args.regime,
        n_meta_steps=args.n_meta_steps,
        tasks_per_step=args.tasks_per_step,
        lr=args.lr,
        steps_available=args.steps_available,
        K=args.k_steps,
        rate_hidden_dim=args.rate_hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        out_path=out_path,
        seed=args.seed,
    )
