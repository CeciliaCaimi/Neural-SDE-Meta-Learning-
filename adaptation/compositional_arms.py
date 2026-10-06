# adaptation/compositional_arms.py
#
# Item 6, phase 3: the four comparison arms for the compositional-
# generalisation experiment, plus a matched-capacity single_z control
# (item 23), behind one uniform interface (build_models / encode / adapt /
# evaluate_task).
#
#   1) single_z                 -- pre-item-1 architecture: ONE latent z
#                                   shared by drift, diffusion and head
#                                   (default z_dim=16; z_dim=32 selectable
#                                   as the item-23 matched-capacity
#                                   control, not one of the 4 main arms).
#                                   Fixed 50-step Adam adaptation (tied: one
#                                   leaf tensor, so Delta z_f == Delta z_g
#                                   by construction) + the original scalar
#                                   gate.
#   2) factorised                -- item 1's (z_f, z_g), the existing fixed
#                                   independent 50-step Adam adaptation
#                                   (adapt_model) + the existing scalar gate,
#                                   reused verbatim from
#                                   adaptation/gated_finetuning_regularized.py
#                                   (both are already latent-dim- and
#                                   theta-type-agnostic, so no duplication
#                                   is needed for this arm).
#   3) factorised_learned_rates  -- item 4's differentiable K-step adaptation,
#                                   adaptation_mode="conditioned" (task-
#                                   conditioned etaf/etag/betaf/betag, no
#                                   gate -- a_f/a_g are not applicable, logged
#                                   as NaN).
#   4) factorised_selective      -- item 4's differentiable K-step adaptation,
#                                   adaptation_mode="conditioned_plus_gate"
#                                   (task-conditioned rates + a fresh
#                                   (a_f, a_g) controller).
#
# This mapping is the literal one in the professor's task list (item 6's
# four bullets map 1:1 onto item 4's four --adaptation-mode variants plus
# the pre-item-1 architecture); see COMPOSITIONAL_EXPERIMENT.md for the
# alternative considered and why this one was kept.
#
# Why a new module instead of editing existing ones: arms 3/4 need a
# MechanismTheta-aware residual feature for the controller input (item 6
# phase 1b: compute_mechanism_error_factorial, not the legacy
# compute_mechanism_error, which only knows the basis-coefficient Theta).
# adaptation/differentiable_adaptation.py's differentiable_adapt() hardcodes
# the legacy compute_mechanism_error -- rather than branch that existing,
# already-relied-upon function (items 1-4's queued real-GPU comparisons use
# it unchanged), this module duplicates its ~30-line body as
# differentiable_adapt_factorial(), swapping only that one call, and reuses
# everything else (k_step_adapt, get_rates, initial_grad_norms,
# build_controller_input) by direct import. Same rationale as
# data_gen/simulate_true_sde.py's simulate_batch_generic /
# evaluation/compositional_metrics.py.
#
# adapt_model/compute_residual (the factorised arm's fixed-Adam adaptation
# and its residual diagnostic) ARE reused directly, unmodified -- both are
# already generic over latent dimension and never touch theta/ground-truth
# at all, so there is nothing MechanismTheta-specific to duplicate for them.

import copy
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from adaptation.differentiable_adaptation import (
    ADAPTATION_MODES,
    DEFAULT_K,
    RateController,
    get_rates,
    initial_grad_norms,
    k_step_adapt,
)
from adaptation.gate_controller import LearnedGateController, build_controller_input
from adaptation.gated_finetuning_regularized import (
    ADAPT_STEPS,
    BETA_REG,
    GATE_ALPHA,
    GATE_TAU,
    LR_HEAD,
    LR_Z,
    MC_SAMPLES,
    adapt_model,
    compute_residual,
)
from data_gen.mechanism_library import MechanismTheta
from dataloaders.trajectory_datasets import apply_scaler_to_trajectories, invert_scaler_to_original
from evaluation.compositional_metrics import compute_mechanism_error_factorial
from models.encoder import TrajEncoder
from models.head import ForecastHead
from models.neural_sde import NeuralSDE
from models.single_latent_encoder import SingleLatentEncoder
from training.train_meta import simulate_neural_sde_batch

ARMS: Tuple[str, ...] = ("single_z", "factorised", "factorised_learned_rates", "factorised_selective")

# Arm -> item-4 differentiable-adaptation mode (arms 1/2 use the fixed
# 50-step Adam path instead, not this mapping).
_ARM_TO_DIFF_MODE: Dict[str, str] = {
    "factorised_learned_rates": "conditioned",
    "factorised_selective": "conditioned_plus_gate",
}


def build_models(
    arm: str,
    x_dim: int,
    z_dim: int,
    encoder_hidden_dim: int,
    sde_hidden_dim: int,
    head_hidden_dim: int,
    device,
    rate_hidden_dim: int = 32,
    gate_hidden_dim: int = 32,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, nn.Module]]:
    """Construct (encoder, sde, head, extra) for one arm.

    `extra` holds the arm-specific, freshly-initialised adaptation-policy
    modules (`rate_controller`, and for factorised_selective also
    `gate_controller`) that train_adapt() meta-trains; empty for
    single_z/factorised (fixed-rate adaptation has no meta-parameters).
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm={arm!r}, expected one of {ARMS}")

    if arm == "single_z":
        encoder = SingleLatentEncoder(x_dim, z_dim, encoder_hidden_dim).to(device)
    else:
        encoder = TrajEncoder(x_dim, z_dim, encoder_hidden_dim).to(device)

    sde = NeuralSDE(x_dim, z_dim, z_dim, sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, z_dim, head_hidden_dim).to(device)

    extra: Dict[str, nn.Module] = {}
    if arm == "factorised_learned_rates":
        extra["rate_controller"] = RateController(z_dim=z_dim, hidden_dim=rate_hidden_dim).to(device)
    elif arm == "factorised_selective":
        extra["rate_controller"] = RateController(z_dim=z_dim, hidden_dim=rate_hidden_dim).to(device)
        extra["gate_controller"] = LearnedGateController(z_dim=z_dim, hidden_dim=gate_hidden_dim).to(device)

    return encoder, sde, head, extra


def encode(arm: str, encoder: nn.Module, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder and return (z_f, z_g).

    For single_z, z_f and z_g are the SAME tensor object -- this is what
    ties them (see module docstring / models/single_latent_encoder.py).
    """
    if arm == "single_z":
        z = encoder(obs)
        return z, z
    return encoder(obs)


# ---------------------------------------------------------------------------
# Fixed 50-step Adam adaptation, tied-latent variant (arm 1: single_z).
# Mirrors adaptation.gated_finetuning_regularized.adapt_model exactly,
# except there is one leaf tensor `z_adapted` used for BOTH the drift and
# diffusion argument slots, so ||Delta z_f|| == ||Delta z_g|| by
# construction (there is only one Delta z).
# ---------------------------------------------------------------------------

def adapt_model_single(sde, head_init, z_init, support, gen, cfg):
    start_time = time.time()
    head = copy.deepcopy(head_init)
    head.train()
    z_adapted = z_init.clone().detach()
    z_adapted.requires_grad = True

    optimizer = optim.Adam([
        {"params": head.parameters(), "lr": LR_HEAD},
        {"params": [z_adapted], "lr": LR_Z},
    ])
    for p in sde.parameters():
        p.requires_grad = False

    B, T, D = support.shape
    T_full = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = T - 1
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs

    grad_norm_z_init = None
    for step_idx in range(ADAPT_STEPS):
        optimizer.zero_grad()
        z_batch = z_adapted.expand(B, -1)

        traj = simulate_neural_sde_batch(sde, support[:, 0], z_batch, z_batch, T_sim, n_sim, x_max, gen)
        valid_len = min(traj.shape[1], T)
        traj_slice = traj[:, :valid_len, :]
        supp_slice = support[:, :valid_len, :]

        loss_path = F.mse_loss(traj_slice, supp_slice)

        final_state_pred = traj_slice[:, -1, :]
        final_state_target = supp_slice[:, -1, :]
        head_pred = head(final_state_pred, z_batch, z_batch)
        loss_head = F.mse_loss(head_pred, final_state_target)

        loss_reg = BETA_REG * torch.sum(z_adapted ** 2)

        total_loss = loss_path + loss_head + loss_reg
        total_loss.backward()
        if step_idx == 0:
            grad_norm_z_init = z_adapted.grad.norm().item()
        optimizer.step()

    return head, z_adapted.detach(), time.time() - start_time, grad_norm_z_init


# ---------------------------------------------------------------------------
# Differentiable K-step adaptation, MechanismTheta-aware variant (arms 3/4).
# See module docstring for why this duplicates
# adaptation.differentiable_adaptation.differentiable_adapt's body instead
# of editing it.
# ---------------------------------------------------------------------------

def differentiable_adapt_factorial(
    encoder, sde, head, support, gen, cfg, theta: MechanismTheta, adaptation_mode: str,
    target_scaler=None, rate_controller=None, global_rates=None, gate_controller=None,
    K: int = DEFAULT_K,
):
    """Same as adaptation.differentiable_adaptation.differentiable_adapt,
    but scores the pre-adaptation residual feature against a MechanismTheta
    via compute_mechanism_error_factorial instead of the legacy Theta-only
    compute_mechanism_error. Additionally returns a_f/a_g (NaN tensors for
    "conditioned", the gate's actual output for "conditioned_plus_gate") so
    callers can log them without recomputing anything.
    """
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(f"Unknown adaptation_mode={adaptation_mode!r}, expected one of {ADAPTATION_MODES}")
    if adaptation_mode == "conditioned_plus_gate" and gate_controller is None:
        raise ValueError("adaptation_mode='conditioned_plus_gate' requires a gate_controller")

    support_in = apply_scaler_to_trajectories(support, target_scaler) if target_scaler is not None else support

    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        zf_all, zg_all = encoder(support_in[:, :enc_len])
        zf_init = zf_all.mean(dim=0, keepdim=True)
        zg_init = zg_all.mean(dim=0, keepdim=True)

    ctrl_input = None
    if adaptation_mode in ("conditioned", "conditioned_plus_gate"):
        grad_norm_zf, grad_norm_zg = initial_grad_norms(sde, head, zf_init, zg_init, support_in, gen, cfg)
        drift_residual, diffusion_residual = compute_mechanism_error_factorial(
            sde, theta, zf_init, zg_init, support_in, support, target_scaler
        )
        ctrl_input = build_controller_input(
            zf_init, zg_init, drift_residual, diffusion_residual, grad_norm_zf, grad_norm_zg,
        )

    eta_f, eta_g, beta_f, beta_g = get_rates(
        adaptation_mode, ctrl_input=ctrl_input, rate_controller=rate_controller,
        global_rates=global_rates, device=zf_init.device, dtype=zf_init.dtype,
    )

    zf_K, zg_K = k_step_adapt(
        sde, head, zf_init, zg_init, support_in, gen, cfg, eta_f, eta_g, beta_f, beta_g, K=K,
    )

    if adaptation_mode == "conditioned_plus_gate":
        a = gate_controller(ctrl_input)  # (1, 2)
        a_f, a_g = a[0, 0], a[0, 1]
        zf_final = zf_init.detach() + a_f * (zf_K - zf_init.detach())
        zg_final = zg_init.detach() + a_g * (zg_K - zg_init.detach())
    else:
        zf_final, zg_final = zf_K, zg_K
        a_f = torch.tensor(float("nan"))
        a_g = torch.tensor(float("nan"))

    return zf_final, zg_final, zf_init.detach(), zg_init.detach(), a_f, a_g


# ---------------------------------------------------------------------------
# Unified per-task evaluation.
# ---------------------------------------------------------------------------

def evaluate_task(
    arm: str,
    encoder: nn.Module,
    sde: nn.Module,
    head: nn.Module,
    extra: Dict[str, nn.Module],
    support: torch.Tensor,
    query: torch.Tensor,
    theta: MechanismTheta,
    gen: torch.Generator,
    cfg,
    target_scaler=None,
    mc_samples: int = MC_SAMPLES,
    K: int = DEFAULT_K,
) -> dict:
    """Run one task's adaptation + evaluation for `arm`, returning a dict
    with every metric item 6 phase 4's CSV needs: rollout_error, drift_error,
    diffusion_error, nll, a_f, a_g, delta_zf_norm, delta_zg_norm (plus
    residual_error/adapt_time diagnostics).

    `support`/`query`: (N, T, D) raw (unnormalized) trajectories.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm={arm!r}, expected one of {ARMS}")

    if target_scaler is not None:
        support_in = apply_scaler_to_trajectories(support, target_scaler)
        query_in = apply_scaler_to_trajectories(query, target_scaler)
    else:
        support_in, query_in = support, query

    T_full, n_steps = cfg.time_grid.T, cfg.time_grid.n_steps
    x_max = cfg.stability.max_state_abs
    B_q = query_in.shape[0]

    if arm in ("single_z", "factorised"):
        with torch.no_grad():
            enc_len = min(support_in.shape[1], 50)
            if arm == "single_z":
                z_all = encoder(support_in[:, :enc_len])
                zf_init = zg_init = z_all.mean(dim=0, keepdim=True)
            else:
                zf_all, zg_all = encoder(support_in[:, :enc_len])
                zf_init = zf_all.mean(dim=0, keepdim=True)
                zg_init = zg_all.mean(dim=0, keepdim=True)

        if arm == "single_z":
            head_opt, z_opt, adapt_time, _ = adapt_model_single(sde, head, zf_init, support_in, gen, cfg)
            zf_opt = zg_opt = z_opt
        else:
            head_opt, zf_opt, zg_opt, adapt_time, _, _ = adapt_model(sde, head, zf_init, zg_init, support_in, gen, cfg)

        delta_zf_norm = (zf_opt - zf_init).norm().item()
        delta_zg_norm = (zg_opt - zg_init).norm().item()

        d_res = compute_residual(sde, head_opt, zf_opt, zg_opt, support_in, gen, cfg)
        data_var = support_in.var().item()
        N = support_in.shape[1]
        d_norm = d_res / (data_var * (N ** 0.5) + 1e-8)
        g = torch.sigmoid(torch.tensor(GATE_ALPHA * (GATE_TAU - d_norm))).item()
        a_f = a_g = float("nan")

        zf_smart = zf_opt.expand(B_q, -1); zf_safe = torch.zeros_like(zf_smart)
        zg_smart = zg_opt.expand(B_q, -1); zg_safe = torch.zeros_like(zg_smart)

        mc_preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                t_smart = simulate_neural_sde_batch(sde, query_in[:, 0], zf_smart, zg_smart, T_full, n_steps, x_max, gen)
                t_safe = simulate_neural_sde_batch(sde, query_in[:, 0], zf_safe, zg_safe, T_full, n_steps, x_max, gen)
                mc_preds.append((1 - g) * t_safe + g * t_smart)
    else:
        adaptation_mode = _ARM_TO_DIFF_MODE[arm]
        zf_opt, zg_opt, zf_init, zg_init, a_f_t, a_g_t = differentiable_adapt_factorial(
            encoder, sde, head, support, gen, cfg, theta, adaptation_mode,
            target_scaler=target_scaler, rate_controller=extra["rate_controller"],
            gate_controller=extra.get("gate_controller"), K=K,
        )
        zf_opt = zf_opt.detach(); zg_opt = zg_opt.detach()
        delta_zf_norm = (zf_opt - zf_init).norm().item()
        delta_zg_norm = (zg_opt - zg_init).norm().item()
        a_f = a_f_t.item() if torch.is_tensor(a_f_t) else a_f_t
        a_g = a_g_t.item() if torch.is_tensor(a_g_t) else a_g_t
        adapt_time = float("nan")  # not wall-clock-comparable to the 50-step Adam path
        d_res = compute_residual(sde, head, zf_opt, zg_opt, support_in, gen, cfg)

        zf_q = zf_opt.expand(B_q, -1); zg_q = zg_opt.expand(B_q, -1)
        mc_preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                mc_preds.append(simulate_neural_sde_batch(sde, query_in[:, 0], zf_q, zg_q, T_full, n_steps, x_max, gen))

    mc_tensor = torch.stack(mc_preds, dim=0)
    mean_norm = mc_tensor.mean(dim=0)
    var_norm = mc_tensor.var(dim=0) + 1e-6

    if target_scaler is not None:
        mean = invert_scaler_to_original(mean_norm, target_scaler)
        query_orig = query
    else:
        mean = mean_norm
        query_orig = query_in

    drift_error, diffusion_error = compute_mechanism_error_factorial(
        sde, theta, zf_opt, zg_opt, query_in, query_orig, target_scaler,
    )

    mse_rollout = F.mse_loss(mean, query_orig).item()
    nll = F.gaussian_nll_loss(mean_norm, query_in, var_norm).item()

    return {
        "rollout_error": mse_rollout,
        "rmse_rollout": mse_rollout ** 0.5,
        "drift_error": drift_error,
        "diffusion_error": diffusion_error,
        "nll": nll,
        "a_f": a_f,
        "a_g": a_g,
        "delta_zf_norm": delta_zf_norm,
        "delta_zg_norm": delta_zg_norm,
        "residual_error": d_res,
        "adapt_time": adapt_time,
    }
