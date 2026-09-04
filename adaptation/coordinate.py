"""Coordinate refinement and the adaptation strategies. Section 8.4, equation (26).

    z*_{y,T} = argmin_z { (1/K_T)·Σ_i E_{t,ε}[ w(t)‖ε − ε̂_{φ,z}(x^T_{i,t}, t)‖² ]
                          + (β_0/K_T)·‖z − z̃_{y,T}‖² }

The 1/K_T scaling is deliberate: it holds the gradient scale of the denoising term steady
across support sizes, while the source prior's relative influence decays as evidence grows.

Note that the signature of refine() contains **no tgt_query**. The red line of A.1,
"no target-query use during meta-test refinement", is therefore inexpressible at the
type level rather than left to anyone remembering it.

The strategies differ **only in initial value and prior centre**; all share one refine():
    target_only         z^enc_T = r_psi(D^s_T)      baseline; also the source-free fallback
    source_reuse        z_S                         no transport
    transport           z_S + Delta_gamma(z_S, c)   this method
    transport_no_refine as above but J = 0          isolates the value of refinement
    zero                0, no refinement            no-adaptation control (document baseline)
    zero_refine         0, with refinement          isolates refinement on its own
    oracle              refined on abundant target  attainable upper bound
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from adaptation.budget import AdaptBudget
from diffusion.forward import q_sample
from diffusion.losses import denoising_loss
from models.score_model import ScoreModel

STRATEGIES = (
    "target_only", "source_reuse", "transport", "transport_no_refine", "zero", "oracle",
    "zero_refine",   # refine starting from z=0 -- distinct from zero; isolates refinement
    # Unconstrained references, to attribute oracle error to the backbone or to the basis
    "full_ft", "full_ft_oracle",
)


@dataclass
class AdaptState:
    z: Tensor
    init_z: Tensor
    strategy: str
    steps_taken: int = 0
    final_loss: float = float("nan")
    model: ScoreModel | None = None      # only full fine-tuning returns a modified copy


def refine(
    model: ScoreModel,
    z_init: Tensor,
    tgt_support: Tensor,
    budget: AdaptBudget,
    prior_center: Tensor | None = None,
    weighting: str = "simple",
    gamma: float = 5.0,
    strategy: str = "?",
) -> AdaptState:
    """Optimise z alone; network parameters stay frozen throughout. Equation (26).

    tgt_support : (K_T, C, H, W) -- the only target data permitted to appear here
    prior_center: prior centre z_tilde; None disables the prior (equivalent to beta_0 = 0)
    """
    k_t = int(tgt_support.shape[0])
    z = z_init.detach().clone().requires_grad_(True)

    # Freeze network parameters, restoring them on exit (else one evaluation freezes the model)
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)

    if budget.steps > 0:
        opt = torch.optim.Adam([z], lr=budget.lr)
        for _ in range(budget.steps):
            # Redraw (t, eps) on the K_T support images each step to average out timestep variance
            reps = max(1, budget.noise_batch // k_t)
            x0 = tgt_support.repeat(reps, *([1] * (tgt_support.dim() - 1)))
            nb = q_sample(model.schedule, x0)
            pred = model.eps_hat(nb.x_t, nb.t, z.unsqueeze(0).expand(x0.shape[0], -1))
            loss = denoising_loss(nb.eps, pred, nb.t, model.schedule, weighting, gamma)
            if prior_center is not None and budget.beta0 > 0:
                loss = loss + (budget.beta0 / k_t) * (z - prior_center).pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        final = float(loss.detach())
    else:
        final = float("nan")

    for p, was in saved:
        p.requires_grad_(was)

    return AdaptState(z=z.detach(), init_z=z_init.detach(), strategy=strategy,
                      steps_taken=budget.steps, final_loss=final)


def full_finetune(
    model: ScoreModel,
    tgt_data: Tensor,
    budget: AdaptBudget,
    weighting: str = "simple",
    gamma: float = 5.0,
    strategy: str = "full_ft",
) -> AdaptState:
    """Full score-network fine-tuning -- the comparison listed in sections 3.1 and 12.1.

    It bypasses the coordinate mechanism (z fixed at 0) and updates every weight. It shows what
    is reachable **without** the low-dimensional constraint, which is how oracle error is attributed.

    The step count J matches the coordinate method; the learning rate is separate (lr_weights).
    """
    import copy

    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()

    n = int(tgt_data.shape[0])
    z0 = torch.zeros(m.k, device=tgt_data.device, dtype=tgt_data.dtype)
    opt = torch.optim.Adam(m.parameters(), lr=budget.lr_weights)
    final = float("nan")
    for _ in range(budget.steps):
        reps = max(1, budget.noise_batch // n)
        x0 = tgt_data.repeat(reps, *([1] * (tgt_data.dim() - 1)))
        nb = q_sample(m.schedule, x0)
        pred = m.eps_hat(nb.x_t, nb.t, z0.unsqueeze(0).expand(x0.shape[0], -1))
        loss = denoising_loss(nb.eps, pred, nb.t, m.schedule, weighting, gamma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = float(loss.detach())
    m.eval()
    return AdaptState(z=z0, init_z=z0, strategy=strategy,
                      steps_taken=budget.steps, final_loss=final, model=m)


# ---------------------------------------------------------------------------
# Strategies -- these fix only the initial value and the prior centre
# ---------------------------------------------------------------------------

def adapt(
    strategy: str,
    model: ScoreModel,
    encoder,
    transport,
    batch,
    budget: AdaptBudget,
    weighting: str = "simple",
    oracle_data: Tensor | None = None,
) -> AdaptState:
    """Dispatch one adaptation by name. Every strategy shares one budget."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}'; expected one of {STRATEGIES}")

    if strategy in ("full_ft", "full_ft_oracle"):
        data = oracle_data if strategy == "full_ft_oracle" else batch.tgt_support
        if data is None:
            raise ValueError(f"strategy '{strategy}' needs oracle_data, which was not supplied")
        return full_finetune(model, data, budget, weighting, strategy=strategy)

    with torch.no_grad():
        z_s = encoder(batch.src_support)
        rel = None if transport.relation_emb is None else batch.relation.reshape(1)
        if strategy in ("transport", "transport_no_refine"):
            init = transport(z_s.unsqueeze(0), rel).squeeze(0)
            center = init
        elif strategy == "target_only":
            init = encoder(batch.tgt_support)
            center = init                      # prior centre is itself; budget and regulariser aligned
        elif strategy == "source_reuse":
            init = z_s
            center = init
        elif strategy in ("zero", "zero_refine"):
            init = torch.zeros(model.k, device=z_s.device, dtype=z_s.dtype)
            center = None
        else:                                   # oracle
            init = transport(z_s.unsqueeze(0), rel).squeeze(0)
            center = None                       # data is abundant, so no prior

    if strategy == "transport_no_refine":
        return AdaptState(z=init, init_z=init, strategy=strategy, steps_taken=0)

    # 'zero' is the "no adaptation" entry of the baseline list: z=0 and **no refinement**.
    # It previously skipped only when budget.steps == 0, silently making it "refine from 0",
    # which contradicted the label. Use 'zero_refine' explicitly for that behaviour.
    if strategy == "zero":
        return AdaptState(z=init, init_z=init, strategy=strategy, steps_taken=0)

    support = oracle_data if strategy == "oracle" else batch.tgt_support

    return refine(model, init, support, budget, center, weighting, strategy=strategy)
