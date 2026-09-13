"""The guardrails of sections 15 and A.5, run automatically from the training loop.

They live in diagnostics/ rather than evaluation/ deliberately: A.5 lists **stopping**
conditions, and reading them only after a run defeats the purpose -- the compute is gone.

Three things:
  1. r_basis = ||eps_res|| / (||eps_base|| + eps) -- is the basis used at all?
  2. correct z vs z=0 vs shuffled z -- is the coordinate used at all?
  3. compare both against the stopping conditions of A.5 and warn when one fires
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from config.base_config import BaseConfig
from diffusion.forward import q_sample
from diffusion.losses import denoising_loss
from episodes.dataset import EpisodeBatch
from models.score_model import ScoreModel
from models.set_encoder import SetEncoder
from models.transport import Transport
from training.meta_train import compute_coordinates


@dataclass
class DiagnosticReport:
    values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)  # E13: per-episode paired diffs

    def format(self) -> str:
        head = "  ".join(f"{k}={v:.4f}" for k, v in self.values.items())
        if not self.warnings:
            return head
        return head + "\n" + "\n".join(f"    ⚠ {w}" for w in self.warnings)


@torch.no_grad()
def run_diagnostics(
    model: ScoreModel,
    encoder: SetEncoder,
    transport: Transport,
    batches: list[EpisodeBatch],
    cfg: BaseConfig,
    n_draws: int = 1,
) -> DiagnosticReport:
    """Run the controls on several episodes and return means. Batches must come from data that did not enter this step's gradient.

    ``n_draws`` (E13): average each episode's controls over this many noise draws, cutting
    the variance of the per-episode paired differences that are logged as ``series``.
    """
    sched = model.schedule
    w, gamma = cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma
    acc: dict[str, list[float]] = {}

    def push(k: str, v: float) -> None:
        acc.setdefault(k, []).append(v)

    if len(batches) < 2:
        raise ValueError("the shuffled control needs at least 2 episodes -- z is shuffled across them")

    # ---- First pass: compute the three coordinates of every episode ----
    # Note: every query image in one episode shares the same z, so **shuffling along the batch
    # dimension is a no-op**. A real shuffled control must use **another episode's** coordinate.
    coords = [compute_coordinates(encoder, transport, b) for b in batches]

    # ---- Has the coordinate collapsed? ----
    z_stack = torch.stack([c[2] for c in coords])                    # (E, k) the z_tilde_T of each episode
    spread = torch.cdist(z_stack, z_stack)
    off_diag = spread[~torch.eye(len(coords), dtype=torch.bool, device=z_stack.device)]
    mean_norm = z_stack.norm(dim=1).mean()
    z_spread_rel = (off_diag.mean() / mean_norm.clamp_min(1e-8)).item()
    # E13: the substitution ("mean") coordinate is the running mean over training episodes
    # -- the z_center EMA buffer -- not the mean of the current diagnostic batch. Substituting
    # it for each episode's own coordinate removes all task-specific content while keeping the
    # common offset, which is what separates real adaptation from the basis acting as shared
    # capacity. Fall back to the batch mean only when the buffer was never populated.
    z_mean = model.z_center if float(model.z_center.norm()) > 0.0 else z_stack.mean(dim=0)

    control_names = ("correct", "zero", "shuffled", "random", "target_only", "mean_z")
    delta_task_series: list[float] = []      # per-episode L(z_mean) - L(z_correct)
    gain_zero_series: list[float] = []       # per-episode L(z=0)   - L(z_correct)
    draws = max(1, int(n_draws))

    # ---- Second pass: the loss under each control, averaged over n_draws draws ----
    for i, (batch, (z_s, z_enc_t, z_tld_t)) in enumerate(zip(batches, coords)):
        n = batch.tgt_query.shape[0]

        def rep(z: Tensor) -> Tensor:
            return z.unsqueeze(0).expand(n, -1)

        z_correct = rep(z_tld_t)
        z_zero = torch.zeros_like(z_correct)
        z_mismatch = rep(coords[(i + 1) % len(coords)][2])           # another episode's coordinate
        z_mean_rep = rep(z_mean)

        ep = {name: 0.0 for name in control_names}
        ep_rbasis = 0.0
        for _ in range(draws):
            nb = q_sample(sched, batch.tgt_query)
            z_rand = torch.randn_like(z_correct)
            z_rand = z_rand / z_rand.norm(dim=1, keepdim=True) * z_correct.norm(dim=1, keepdim=True)
            preds = model.eps_hat_many(
                nb.x_t, nb.t, [z_correct, z_zero, z_mismatch, z_rand, rep(z_enc_t), z_mean_rep])
            for name, pred in zip(control_names, preds):
                ep[name] += denoising_loss(nb.eps, pred, nb.t, sched, w, gamma).item()
            ep_rbasis += model.basis_usage(nb.x_t, nb.t, z_correct).mean().item()

        for name in control_names:
            ep[name] /= draws
            push(f"loss_{name}", ep[name])
        push("r_basis", ep_rbasis / draws)
        push("|dz_transport|", (z_tld_t - z_s).norm().item())
        # are the source and target coordinates separated at all (relative to their norm)
        push("z_s_vs_enc_rel", ((z_s - z_enc_t).norm() / z_s.norm().clamp_min(1e-8)).item())
        delta_task_series.append(ep["mean_z"] - ep["correct"])
        gain_zero_series.append(ep["zero"] - ep["correct"])

    values = {k: float(sum(v) / len(v)) for k, v in acc.items()}
    values["z_spread_rel"] = z_spread_rel
    values["gain_vs_zero"] = values["loss_zero"] - values["loss_correct"]
    values["gain_vs_shuffled"] = values["loss_shuffled"] - values["loss_correct"]
    values["gain_vs_target_only"] = values["loss_target_only"] - values["loss_correct"]
    values["gain_vs_mean_z"] = values["loss_mean_z"] - values["loss_correct"]
    # E13: delta_task is the task-specific benefit -- the correct coordinate against the
    # running-mean coordinate. gain_vs_mean_z is kept as its alias (a plotting script reads it).
    values["delta_task"] = values["gain_vs_mean_z"]
    values["z_center_norm"] = float(model.z_center.norm())
    # What fraction of the coordinate's whole benefit is actually task-specific?
    # Near 0 means the basis is a constant offset dressed up as adaptation.
    denom = values["gain_vs_zero"]
    values["task_specific_frac"] = (
        values["gain_vs_mean_z"] / denom if abs(denom) > 1e-9 else 0.0)

    series = {"delta_task": delta_task_series, "gain_vs_zero": gain_zero_series}
    return DiagnosticReport(
        values=values,
        warnings=_stop_conditions(values, center_coords=getattr(model, "center_coords", False)),
        series=series,
    )


def _stop_conditions(v: dict[str, float], collapse_tol: float = 0.05,
                     center_coords: bool = False) -> list[str]:
    """A.5: when one of these fires, stop and diagnose rather than scale up."""
    out = []
    if v["r_basis"] < 1e-3:
        out.append(f"r_basis={v['r_basis']:.2e} is near 0 -- the basis is barely used (headline failure of 15)")
    if center_coords:
        # E13: under centring z=0 is an off-centre perturbation, not the null; the null is
        # z = z_center, so delta_task (correct vs the running mean) is the task-specific signal.
        if v.get("delta_task", 0.0) <= 0:
            out.append(
                "delta_task <= 0 under centring -- the correct coordinate does not beat its "
                "own running mean, so there is no task-specific benefit")
    else:
        if v["gain_vs_zero"] <= 0:
            out.append("z=0 matches or beats the correct coordinate -- low-dimensional structure unproven")
        if v.get("gain_vs_zero", 0.0) > 1e-3 and v.get("task_specific_frac", 1.0) < 0.05:
            out.append(
                f"gain_vs_zero={v['gain_vs_zero']:.4f} looks healthy but only "
                f"{100*v['task_specific_frac']:.1f}% of it is task-specific: one shared mean "
                "coordinate does just as well, so the basis is a constant offset, not adaptation")

    # Check collapse first: when collapsed gain_vs_shuffled is ~0, but the cause is the encoder
    collapsed = v["z_spread_rel"] < collapse_tol
    if collapsed:
        out.append(
            f"coordinate collapse: relative spacing of z across episodes is only {v['z_spread_rel']:.3f}"
            f" (threshold {collapse_tol}) -- the encoder emits a near-constant, so"
            " gain_vs_zero reflects the basis acting as shared capacity, not task specificity"
        )
    if v["z_s_vs_enc_rel"] < collapse_tol:
        out.append(
            f"z_S and z^enc_T are only {v['z_s_vs_enc_rel']:.3f} apart relatively -- "
            "the encoder is not separating the source set from the target set"
        )
    if v["gain_vs_shuffled"] <= 0 and not collapsed:
        out.append("shuffling coordinates across episodes changes little -- z does not track the task")
    if v["|dz_transport|"] < 1e-4:
        out.append("Delta_gamma displacement is ~0 -- transport has degenerated to the identity")
    return out
