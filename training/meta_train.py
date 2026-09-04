"""One meta-training step. Section 8.5, equations (27) to (30).

    z_S     = r_ψ(D^s_{y,S})                                       (23)
    z^enc_T = r_ψ(D^s_{y,T})                                       (24)
    z̃_T     = z_S + Δγ(z_S, c_{S→T})                               (25)

    L_src   = E_{x₀ ∈ D^q_{y,S}} [ w(t)‖ε − ε̂_{φ,z_S}‖² ]          (27)
    L_tgt   = E_{x₀ ∈ D^q_{y,T}} [ w(t)‖ε − ε̂_{φ,z^enc_T}‖² ]      (28)
    L_trans = E_{x₀ ∈ D^q_{y,T}} [ w(t)‖ε − ε̂_{φ,z̃_T}‖² ]         (29)

    L_meta  = L_src + λ_T·L_tgt + λ_tr·L_trans + λ_z·(‖z_S‖² + ‖z^enc_T‖²)   (30)

Points to note:
  - Support sets enter only the encoder; query sets are used only for the losses.
  - L_tgt and L_trans act on the **same** target-query batch (a paired comparison, lower
    variance), sharing one noising (common random numbers) and one backbone pass.
  - **v1 never differentiates through refinement**; Delta_gamma is trained by L_trans only.
  - z_tilde_T is excluded from lambda_z: equation (30) regularises only z_S and z^enc_T.
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


@dataclass
class MetaOutputs:
    loss: Tensor
    z_s: Tensor
    z_enc_t: Tensor
    z_tld_t: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


def compute_coordinates(
    encoder: SetEncoder, transport: Transport, batch: EpisodeBatch
) -> tuple[Tensor, Tensor, Tensor]:
    """Where the three coordinates come from. The transport input holds **no target image**."""
    z_s = encoder(batch.src_support)                                    # (k,)
    z_enc_t = encoder(batch.tgt_support)                                # (k,)
    # With a single relation the Transport has no embedding, so no relation is passed (A.2)
    rel = None if transport.relation_emb is None else batch.relation.reshape(1)
    z_tld_t = transport(z_s.unsqueeze(0), rel).squeeze(0)
    return z_s, z_enc_t, z_tld_t


def meta_step(
    model: ScoreModel,
    encoder: SetEncoder,
    transport: Transport,
    batch: EpisodeBatch,
    cfg: BaseConfig,
) -> MetaOutputs:
    sched = model.schedule
    w, gamma = cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma

    z_s, z_enc_t, z_tld_t = compute_coordinates(encoder, transport, batch)

    # ---- Source branch: L_src on D^q_{y,S} with z_S ----
    nb_s = q_sample(sched, batch.src_query)
    n_s = batch.src_query.shape[0]
    eps_s = model.eps_hat(nb_s.x_t, nb_s.t, z_s.unsqueeze(0).expand(n_s, -1))
    l_src = denoising_loss(nb_s.eps, eps_s, nb_s.t, sched, w, gamma)

    # ---- Target branch: L_tgt and L_trans on the same D^q_{y,T} batch ----
    nb_t = q_sample(sched, batch.tgt_query)
    n_t = batch.tgt_query.shape[0]
    eps_tgt, eps_trans = model.eps_hat_many(
        nb_t.x_t, nb_t.t,
        [z_enc_t.unsqueeze(0).expand(n_t, -1), z_tld_t.unsqueeze(0).expand(n_t, -1)],
    )
    l_tgt = denoising_loss(nb_t.eps, eps_tgt, nb_t.t, sched, w, gamma)
    l_trans = denoising_loss(nb_t.eps, eps_trans, nb_t.t, sched, w, gamma)

    # ---- Coordinate regularisation; equation (30) covers only z_S and z^enc_T ----
    l_z = z_s.pow(2).sum() + z_enc_t.pow(2).sum()

    t = cfg.train
    loss = l_src + t.lambda_tgt * l_tgt + t.lambda_trans * l_trans + t.lambda_z * l_z

    with torch.no_grad():
        metrics = {
            "L_meta": loss.item(),
            "L_src": l_src.item(),
            "L_tgt": l_tgt.item(),
            "L_trans": l_trans.item(),
            "L_z": l_z.item(),
            # advantage of transport over target-only; > 0 means transport is better here
            "trans_gain": (l_tgt - l_trans).item(),
            "|z_S|": z_s.norm().item(),
            "|z_enc_T|": z_enc_t.norm().item(),
            # how far transport moves the coordinate; a constant 0 means Delta_gamma is the identity
            "|dz_transport|": (z_tld_t - z_s).norm().item(),
            # agreement with the sparse target encoding; observation only, not a training signal
            "|z_tld-z_enc|": (z_tld_t - z_enc_t).norm().item(),
            # A.1 requires the source and target sample counts to be recorded separately
            "M_S": float(batch.src_support.shape[0]),
            "K_T": float(batch.tgt_support.shape[0]),
            "n_src_query": float(batch.src_query.shape[0]),
            "n_tgt_query": float(batch.tgt_query.shape[0]),
        }
    return MetaOutputs(loss=loss, z_s=z_s, z_enc_t=z_enc_t, z_tld_t=z_tld_t, metrics=metrics)
