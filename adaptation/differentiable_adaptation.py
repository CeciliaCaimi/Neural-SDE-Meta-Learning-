# adaptation/differentiable_adaptation.py
#
# Item 4: meta-learn the adaptation process itself.
#
# Items 1/2/3 fixed the *shape* of test-time adaptation (independent z_f/z_g,
# a real identification objective, an optional learned (a_f, a_g) reweighting
# of an already-computed adaptation delta) but always treated the inner
# adaptation loop itself as a black box: ADAPT_STEPS=50 Adam steps with fixed
# hyperparameters (LR_Z, BETA_REG), and item 3's controller explicitly
# detaches the resulting deltas before using them (see
# adaptation/gate_controller.py / training/train_controller.py's "Design
# choice" note).
#
# This module opens that black box: a short, explicitly differentiable K-step
# gradient-descent adaptation loop over (z_f, z_g), where the per-component
# learning rates and regularisation strengths (eta_f, eta_g, beta_f, beta_g)
# are themselves meta-parameters, optimised end to end by differentiating
# THROUGH the K steps (unlike item 3's detached deltas). This is strictly
# additional: the existing 50-step Adam path (adapt_model), the scalar gate,
# and item 3's detached LearnedGateController are untouched and remain the
# default at test time.
#
# --- The update rule -----------------------------------------------------
# adapt_model()'s existing regularisation term is
#     loss_reg = BETA_REG * (sum(z_f**2) + sum(z_g**2))
# i.e. a single shared constant, pulling both latents toward the origin, with
# a positive weight added to the loss. Item 4 asks for the same sign
# convention -- pull toward a reference point with a positive per-component
# weight -- but toward the *initial* (encoder) latent instead of the origin,
# and with independent weights per component. Rather than encode that as an
# extra loss term inside an autograd-tracked "total loss" (which would need
# an extra factor of 1/2 to reproduce the literal update rule below), each
# step directly computes:
#
#     g_f = grad_{z_f}(support_loss(z_f, z_g))
#     g_g = grad_{z_g}(support_loss(z_f, z_g))
#     z_f <- z_f - eta_f * (g_f + beta_f * (z_f - z_f(0)))
#     z_g <- z_g - eta_g * (g_g + beta_g * (z_g - z_g(0)))
#
# where support_loss = path MSE + endpoint/head MSE, matching adapt_model's
# loss_path + loss_head exactly (just without its separate loss_reg term,
# which is now folded into the explicit update above instead of the loss the
# gradient is taken of). z_f(0), z_g(0) are the frozen encoder outputs and
# never change across the K steps.
#
# This is implemented with torch.autograd.grad(..., create_graph=True) and
# out-of-place tensor arithmetic for the update -- no torch.optim, no
# in-place ops, no stray .detach() calls -- so the graph from z_f/z_g all the
# way back to (eta_f, eta_g, beta_f, beta_g) survives K steps intact. See
# k_step_adapt()'s docstring and smoke_test_diff_adaptation.py for how this
# is verified.
#
# --- Choice of K -----------------------------------------------------------
# The spec asks for K in [5, 10]. We use DEFAULT_K = 8: large enough that the
# latents move measurably (a 2-3 step unroll barely differs from the initial
# point) while keeping the create_graph=True second-order unroll -- which
# retains the full forward computation graph of K full trajectory
# simulations through the (frozen) SDE, one on top of the other -- within a
# manageable memory/compute budget for a first real GPU run. Nothing about
# the implementation is tied to this value; K is a plain function argument.
#
# --- The four comparison variants ------------------------------------------
# All four run the exact same k_step_adapt() loop on the exact same frozen
# encoder/SDE/head; they differ only in where (eta_f, eta_g, beta_f, beta_g)
# come from, and variant (d) additionally reweights the K-step result with a
# fresh (a_f, a_g) gate:
#   (a) fixed      -- hardcoded constants (FIXED_ETA_F etc.), never meta-
#                      trained. Pure control: confirms the K-step loop alone,
#                      with no learned adaptation policy, still runs.
#   (b) global      -- 4 learnable scalars (GlobalRates), not conditioned on
#                      the task, meta-trained via post-adaptation query loss.
#   (c) conditioned -- a small MLP (RateController) mapping the same pooled
#                      support representation item 3's controller uses
#                      (build_controller_input: zf_init, zg_init,
#                      drift/diffusion residual, grad-norm-at-init) to the 4
#                      positive rates, meta-trained the same way.
#   (d) conditioned_plus_gate -- (c) plus a FRESH LearnedGateController
#                      instance (same architecture as item 3, but newly
#                      initialised and trained jointly here -- item 3's saved
#                      weights were trained against a detached 50-step loop
#                      and are not reusable against this differentiable
#                      K-step loop), applied on top of the K-step-adapted
#                      latents exactly as item 3 applies it:
#                          z_f' = z_f(0) + a_f * (z_f_K - z_f(0))
#                          z_g' = z_g(0) + a_g * (z_g_K - z_g(0))
#
# See training/train_diff_adaptation.py for the meta-training loop that
# selects between these via --adaptation-mode, and this module's docstrings
# for exactly which parameters get gradients in each case.
#
# --- Explicit non-goal -------------------------------------------------
# We do NOT implement a recurrent/LSTM-based learned optimiser here. The
# task list is explicit that this is only justified once fixed/global/
# conditioned/conditioned_plus_gate plateau at real scale -- which requires
# real GPU results that do not exist yet (this module ships with a toy-scale
# gradient-flow smoke test only, not a training-quality comparison).

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP
from training.train_meta import simulate_neural_sde_batch
from adaptation.gate_controller import LearnedGateController, build_controller_input
from adaptation.gated_finetuning_regularized import compute_mechanism_error

# Number of differentiable inner-loop adaptation steps (see module docstring
# for why 8, within the spec's [5, 10] range).
DEFAULT_K = 8

# Variant (a) fixed-rate constants -- same order of magnitude as the existing
# 50-step Adam path's LR_Z=1e-2 / BETA_REG=0.01 (adapt_model in
# gated_finetuning_regularized.py), just renamed per-component. These are
# plain Python floats, never wrapped in nn.Parameter, by construction: they
# cannot be meta-trained even by accident.
FIXED_ETA_F = 1e-2
FIXED_ETA_G = 1e-2
FIXED_BETA_F = 0.01
FIXED_BETA_G = 0.01

ADAPTATION_MODES = ("fixed", "global", "conditioned", "conditioned_plus_gate")


def _support_loss(sde, head, zf, zg, support_in, gen, cfg):
    """Path MSE + endpoint/head MSE on the support set, for one (zf, zg).

    Structurally identical to adapt_model()'s loss_path + loss_head in
    gated_finetuning_regularized.py, minus its loss_reg term (regularisation
    toward z(0) is applied explicitly in the k_step_adapt update rule
    instead -- see this module's docstring for why).
    """
    B, T, D = support_in.shape
    T_full = cfg.time_grid.T
    n_steps = cfg.time_grid.n_steps
    dt = T_full / n_steps
    n_sim = T - 1
    T_sim = dt * n_sim
    x_max = cfg.stability.max_state_abs

    zf_batch = zf.expand(B, -1)
    zg_batch = zg.expand(B, -1)

    traj = simulate_neural_sde_batch(sde, support_in[:, 0], zf_batch, zg_batch, T_sim, n_sim, x_max, gen)
    valid_len = min(traj.shape[1], T)
    traj_slice = traj[:, :valid_len, :]
    supp_slice = support_in[:, :valid_len, :]

    loss_path = F.mse_loss(traj_slice, supp_slice)

    final_state_pred = traj_slice[:, -1, :]
    final_state_target = supp_slice[:, -1, :]
    head_pred = head(final_state_pred, zf_batch, zg_batch)
    loss_head = F.mse_loss(head_pred, final_state_target)

    return loss_path + loss_head


def initial_grad_norms(sde, head, zf_init, zg_init, support_in, gen, cfg):
    """Gradient norm of the support loss w.r.t. (z_f, z_g) at the initial
    point, as a plain (detached) diagnostic -- the same quantity item 3's
    controller conditions on (grad_norm_zf_init/grad_norm_zg_init), computed
    here via one throwaway forward/backward (no create_graph) rather than
    reused from the K-step loop, so the diagnostic never entangles with the
    differentiable loop's own graph.
    """
    zf = zf_init.clone().detach().requires_grad_(True)
    zg = zg_init.clone().detach().requires_grad_(True)
    loss = _support_loss(sde, head, zf, zg, support_in, gen, cfg)
    grad_zf, grad_zg = torch.autograd.grad(loss, [zf, zg])
    return grad_zf.norm().item(), grad_zg.norm().item()


def k_step_adapt(sde, head, zf_init, zg_init, support_in, gen, cfg, eta_f, eta_g, beta_f, beta_g, K=DEFAULT_K):
    """Differentiable K-step gradient-descent adaptation of (z_f, z_g).

    Args:
        zf_init, zg_init: (1, z_dim) pooled encoder outputs. Used as both the
            starting point and the fixed z(0) reference in the
            regularisation term; detached internally so no gradient flows
            back through the (frozen) encoder.
        eta_f, eta_g, beta_f, beta_g: positive scalar tensors. These are the
            meta-parameters (or fixed constants) under test -- see module
            docstring for the four variants. They MUST be tensors that carry
            gradient history back to whatever produced them (an
            nn.Parameter, or an MLP's output) for meta-training to work;
            plain Python floats work too (variant (a)) and simply mean no
            gradient is meta-trainable.
        K: number of inner steps (see module docstring for choice of 8).

    Returns:
        zf_K, zg_K: (1, z_dim) tensors after K steps. These remain connected
        to the autograd graph of (eta_f, eta_g, beta_f, beta_g) -- and, if
        those came from a network, to that network's parameters -- so a
        downstream query loss computed from zf_K/zg_K can backprop into the
        rate-conditioning meta-parameters. They do NOT carry gradient back to
        the (frozen) sde/head parameters or the encoder, since those are
        never part of the autograd.grad `inputs` list and sde/head
        parameters have requires_grad=False throughout this module's use.

    Graph-safety notes (see module docstring's "update rule" section and
    smoke_test_diff_adaptation.py for how this is verified end to end):
      - torch.autograd.grad(..., create_graph=True) is used instead of
        loss.backward(), so the returned gradient tensors are themselves
        differentiable w.r.t. zf/zg (and transitively w.r.t. eta/beta).
      - Every update is an out-of-place `z = z - eta * (...)` producing a
        brand new tensor each step -- never `z -= ...` / `z.data = ...` /
        an in-place optimiser step, any of which would sever the graph
        silently (no error, just a zero gradient at the meta-parameters).
      - No `.detach()` is called on zf/zg/eta/beta/grad_zf/grad_zg anywhere
        in this loop (zf_init/zg_init ARE detached once, up front, exactly
        once, before the loop starts -- that detach is intentional: it
        stops gradient flowing into the frozen encoder, not into eta/beta).
    """
    zf0 = zf_init.detach()
    zg0 = zg_init.detach()
    zf = zf0.clone().requires_grad_(True)
    zg = zg0.clone().requires_grad_(True)

    for _ in range(K):
        loss = _support_loss(sde, head, zf, zg, support_in, gen, cfg)
        grad_zf, grad_zg = torch.autograd.grad(loss, [zf, zg], create_graph=True)
        zf = zf - eta_f * (grad_zf + beta_f * (zf - zf0))
        zg = zg - eta_g * (grad_zg + beta_g * (zg - zg0))

    return zf, zg


class GlobalRates(nn.Module):
    """Variant (b): 4 learnable scalars, not conditioned on the task.

    Parameterised as softplus(raw) + 1e-3 for positivity, matching
    NeuralSDE.g's existing softplus(raw_sigma) + 1e-3 diffusion positivity
    convention (models/neural_sde.py) rather than inventing a new one.
    """

    def __init__(self):
        super().__init__()
        init_vals = torch.tensor([FIXED_ETA_F, FIXED_ETA_G, FIXED_BETA_F, FIXED_BETA_G])
        # Inverse-softplus so the *initial* forward() output matches the
        # variant-(a) fixed constants -- a convenient, neutral starting
        # point for meta-training, not a constraint (these 4 values are
        # free to move away from it).
        raw_init = torch.log(torch.expm1(init_vals - 1e-3))
        self.raw = nn.Parameter(raw_init)

    def forward(self):
        """Returns a (4,) tensor: (eta_f, eta_g, beta_f, beta_g)."""
        return F.softplus(self.raw) + 1e-3


class RateController(nn.Module):
    """Variants (c)/(d): MLP mapping the pooled support representation to
    positive (eta_f, eta_g, beta_f, beta_g).

    Reuses adaptation.gate_controller.build_controller_input verbatim as the
    input featurisation -- the same (2*z_dim + 4)-dim pooled support-set
    representation item 3's LearnedGateController conditions on (zf_init,
    zg_init, drift/diffusion residual, grad-norm-at-init) -- rather than
    building a second, parallel featurisation.
    """

    def __init__(self, z_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        input_dim = 2 * z_dim + 4
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=4,
            use_layernorm=False,
        )

    def forward(self, controller_input: torch.Tensor) -> torch.Tensor:
        """controller_input: (batch, 2*z_dim+4) -> (batch, 4) = (eta_f, eta_g, beta_f, beta_g)."""
        raw = self.net(controller_input)
        return F.softplus(raw) + 1e-3


def get_rates(adaptation_mode, ctrl_input=None, rate_controller=None, global_rates=None, device=None, dtype=None):
    """Return (eta_f, eta_g, beta_f, beta_g) as 0-d tensors for the given mode.

    ctrl_input is required (and rate_controller must be provided) for
    "conditioned"/"conditioned_plus_gate"; global_rates must be provided for
    "global". "fixed" needs neither -- it returns the hardcoded constants as
    plain tensors with no gradient history, by construction.
    """
    if adaptation_mode == "fixed":
        mk = lambda v: torch.tensor(v, device=device, dtype=dtype)
        return mk(FIXED_ETA_F), mk(FIXED_ETA_G), mk(FIXED_BETA_F), mk(FIXED_BETA_G)
    if adaptation_mode == "global":
        if global_rates is None:
            raise ValueError("adaptation_mode='global' requires a GlobalRates instance")
        rates = global_rates()
        return rates[0], rates[1], rates[2], rates[3]
    if adaptation_mode in ("conditioned", "conditioned_plus_gate"):
        if rate_controller is None or ctrl_input is None:
            raise ValueError(f"adaptation_mode={adaptation_mode!r} requires a RateController and ctrl_input")
        rates = rate_controller(ctrl_input)
        return rates[0, 0], rates[0, 1], rates[0, 2], rates[0, 3]
    raise ValueError(f"Unknown adaptation_mode={adaptation_mode!r}, expected one of {ADAPTATION_MODES}")


def differentiable_adapt(
    encoder, sde, head, support, gen, cfg, theta, adaptation_mode,
    target_scaler=None, rate_controller=None, global_rates=None, gate_controller=None,
    K=DEFAULT_K,
):
    """Run item 4's full per-task adaptation: pool support -> (rates) ->
    K-step differentiable adaptation -> optional (a_f, a_g) gate.

    Args:
        support: (N_shots, T, D) raw (unnormalized) support trajectories.
        theta: ground-truth Theta for this task, needed by "conditioned"/
            "conditioned_plus_gate" to compute the pre-adaptation drift/
            diffusion residual feature (same as item 3).
        adaptation_mode: one of ADAPTATION_MODES.
        rate_controller: required for "conditioned"/"conditioned_plus_gate".
        global_rates: required for "global".
        gate_controller: required for "conditioned_plus_gate" -- a FRESH
            LearnedGateController instance (see module docstring for why it
            must not be item 3's saved checkpoint).

    Returns:
        zf_final, zg_final: (1, z_dim), connected to the meta-parameters'
            autograd graph (see k_step_adapt).
        zf_init, zg_init: (1, z_dim), detached -- the pre-adaptation pooled
            encoder latents, for diagnostics/logging.
    """
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(f"Unknown adaptation_mode={adaptation_mode!r}, expected one of {ADAPTATION_MODES}")
    if adaptation_mode == "conditioned_plus_gate" and gate_controller is None:
        raise ValueError("adaptation_mode='conditioned_plus_gate' requires a gate_controller")

    from dataloaders.trajectory_datasets import apply_scaler_to_trajectories

    if target_scaler is not None:
        support_in = apply_scaler_to_trajectories(support, target_scaler)
    else:
        support_in = support

    with torch.no_grad():
        enc_len = min(support_in.shape[1], 50)
        zf_all, zg_all = encoder(support_in[:, :enc_len])
        zf_init = zf_all.mean(dim=0, keepdim=True)
        zg_init = zg_all.mean(dim=0, keepdim=True)

    ctrl_input = None
    if adaptation_mode in ("conditioned", "conditioned_plus_gate"):
        grad_norm_zf, grad_norm_zg = initial_grad_norms(sde, head, zf_init, zg_init, support_in, gen, cfg)
        drift_residual, diffusion_residual = compute_mechanism_error(
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
        a = gate_controller(ctrl_input)  # (1, 2) in [0, 1]
        a_f, a_g = a[0, 0], a[0, 1]
        zf_final = zf_init.detach() + a_f * (zf_K - zf_init.detach())
        zg_final = zg_init.detach() + a_g * (zg_K - zg_init.detach())
    else:
        zf_final, zg_final = zf_K, zg_K

    return zf_final, zg_final, zf_init.detach(), zg_init.detach()
