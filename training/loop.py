"""Training loop: EMA, optimiser, checkpoints, deterministic seeding, throughput logging.

Independent of any particular experiment -- changing domain or backbone reuses this loop.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from config.base_config import BaseConfig
from diagnostics.controls import run_diagnostics
from diffusion.schedule import NoiseSchedule
from domains.cifar100 import load_cifar100
from episodes.dataset import EpisodeLoader
from episodes.splits import load_split
from models.backbone import build_backbone
from models.score_model import ScoreModel
from models.set_encoder import SetEncoder
from models.transport import Transport
from training.meta_train import meta_step

import models.unet  # noqa: F401  -- triggers the @register_backbone registration


class EMA:
    """Exponential moving average of the parameters. A.2 lists EMA as a default."""

    def __init__(self, modules: list[nn.Module], decay: float = 0.999) -> None:
        self.decay = decay
        self.modules = modules
        self.shadow = [
            {k: v.detach().clone().float() for k, v in m.state_dict().items()
             if v.is_floating_point()}
            for m in modules
        ]

    @torch.no_grad()
    def update(self) -> None:
        for m, sh in zip(self.modules, self.shadow):
            for k, v in m.state_dict().items():
                if k in sh:
                    sh[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, modules: list[nn.Module]) -> None:
        for m, sh in zip(modules, self.shadow):
            sd = m.state_dict()
            for k, v in sh.items():
                sd[k].copy_(v)

    @contextlib.contextmanager
    def applied(self):
        """Temporarily swap in the EMA weights, restoring them exactly on exit.

        A.2 makes EMA a default, so evaluation and diagnostics should both read EMA weights;
        otherwise the reported numbers come from different parameters than the checkpoint.
        """
        backup = [
            {k: v.detach().clone() for k, v in m.state_dict().items() if v.is_floating_point()}
            for m in self.modules
        ]
        try:
            self.copy_to(self.modules)
            yield
        finally:
            with torch.no_grad():
                for m, bk in zip(self.modules, backup):
                    sd = m.state_dict()
                    for k, v in bk.items():
                        sd[k].copy_(v)


def build(cfg: BaseConfig, device: torch.device) -> tuple[ScoreModel, SetEncoder, Transport]:
    """Assemble the three modules from the config. The backbone is named by string alone."""
    sched = NoiseSchedule(cfg.diffusion.n_steps, cfg.diffusion.schedule)
    backbone = build_backbone(cfg.model.backbone, **cfg.model.backbone_kwargs)
    model = ScoreModel(backbone, sched, k=cfg.model.k,
                       basis_init_scale=cfg.model.basis_init_scale).to(device)
    encoder = SetEncoder(
        image_channels=backbone.spec.image_channels, image_size=backbone.spec.image_size,
        width=cfg.model.encoder_width, feature_dim=cfg.model.encoder_feature_dim,
        k=cfg.model.k, hidden=cfg.model.encoder_hidden,
        pooling=cfg.model.encoder_pooling,
    ).to(device)
    transport = Transport(
        k=cfg.model.k, n_relations=cfg.model.n_relations,
        relation_dim=cfg.model.relation_dim, hidden=cfg.model.transport_hidden,
        out_moments=cfg.model.transport_out_moments,
    ).to(device)
    return model, encoder, transport


def save_checkpoint(path: str, step: int, cfg: BaseConfig, model, encoder, transport,
                    opt, ema: EMA) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "model": model.state_dict(),
            "encoder": encoder.state_dict(),
            "transport": transport.state_dict(),
            "optimizer": opt.state_dict(),
            "ema": ema.shadow,
        },
        path,
    )


def train(cfg: BaseConfig) -> str:
    torch.manual_seed(cfg.global_seed)
    np.random.seed(cfg.global_seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    raw = load_cifar100()
    if cfg.episodes.scheme == "domainshift":
        from episodes.domainshift import load_domainshift, DomainShiftLoader
        split = load_domainshift(cfg.episodes.domainshift_path)
        # The relation is the corruption type. More than one enables the relation embedding:
        # indexing by corruption is legitimate, because corruptions recur across all splits.
        if getattr(cfg.episodes, "severity_override", None):
            from dataclasses import replace as _replace
            split = _replace(split, config=_replace(
                split.config, severity=cfg.episodes.severity_override))
        n_cor = len(split.config.corruptions)
        cfg.model.n_relations = n_cor if n_cor > 1 else None
        loader = DomainShiftLoader(
            raw, split, "train", device=device,
            enc_source_images=cfg.episodes.enc_source_images,
            query_batch=cfg.episodes.query_batch,
            k_shots=cfg.episodes.k_shots, seed=cfg.global_seed)
        diag_loader = DomainShiftLoader(
            raw, split, "val", device=device,
            enc_source_images=cfg.episodes.enc_source_images,
            query_batch=cfg.episodes.query_batch,
            k_shots=cfg.episodes.k_shots, seed=cfg.global_seed + 1)
        diag_loader.images = loader.images
    else:
        split = load_split(cfg.episodes.split_path)
        loader = EpisodeLoader(
            raw, split, "train", device=device,
            enc_source_images=cfg.episodes.enc_source_images,
            query_batch=cfg.episodes.query_batch,
            k_shots=cfg.episodes.k_shots, seed=cfg.global_seed,
        )
        # Held-out diagnostics use the **val** split, forward only.
        # Protocol: test is for final reporting; no reading during training may come from it.
        diag_loader = EpisodeLoader(
            raw, split, "val", device=device,
            enc_source_images=cfg.episodes.enc_source_images,
            query_batch=cfg.episodes.query_batch,
            k_shots=cfg.episodes.k_shots, seed=cfg.global_seed + 1,
            pin_to_device=False,
        )
        diag_loader.images = loader.images          # reuse the one resident copy of the images
        diag_loader._on_device = True

    model, encoder, transport = build(cfg, device)
    modules = [model, encoder, transport]
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    ema = EMA(modules, cfg.train.ema_decay)

    n = model.n_parameters()
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.train.ckpt_dir, f"{cfg.run_name}_log.jsonl")
    print(f"device {device} | backbone {cfg.model.backbone} | k={cfg.model.k}")
    print(f"φ {n['total_phi']/1e6:.2f}M + encoder {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M"
          f" + transport {sum(p.numel() for p in transport.parameters())/1e3:.1f}K")
    print(f"training episode combinations {len(loader)} | {cfg.train.steps} steps\n")

    use_amp = cfg.train.amp and device.type == "cuda"
    t0 = time.time()
    for m in modules:
        m.train()

    with open(log_path, "w", encoding="utf-8") as logf:
        for step in range(1, cfg.train.steps + 1):
            # Linear warmup
            lr = cfg.train.lr * min(1.0, step / max(1, cfg.train.warmup_steps))
            for g in opt.param_groups:
                g["lr"] = lr

            batch = loader.sample()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = meta_step(model, encoder, transport, batch, cfg)

            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()
            ema.update()

            if step % cfg.train.log_every == 0 or step == 1:
                m = out.metrics | {"step": step, "lr": lr, "grad_norm": float(gnorm),
                                   "sec": time.time() - t0}
                logf.write(json.dumps(m) + "\n")
                logf.flush()
                print(f"step {step:>6}  L={m['L_meta']:.4f}  src={m['L_src']:.4f}  "
                      f"tgt={m['L_tgt']:.4f}  trans={m['L_trans']:.4f}  "
                      f"gain={m['trans_gain']:+.4f}  |dz|={m['|dz_transport|']:.3f}  "
                      f"{step/(time.time()-t0):.1f} it/s")

            if step % cfg.train.diagnose_every == 0:
                for mm in modules:
                    mm.eval()
                with ema.applied():                       # diagnostics read EMA weights
                    rep = run_diagnostics(model, encoder, transport,
                                          diag_loader.sample_many(8), cfg)
                for mm in modules:
                    mm.train()
                print(f"  [diagnostics @ {step}] {rep.format()}")
                logf.write(json.dumps({"step": step, "diagnostics": rep.values,
                                       "warnings": rep.warnings}) + "\n")
                logf.flush()

            if step % cfg.train.ckpt_every == 0 or step == cfg.train.steps:
                p = os.path.join(cfg.train.ckpt_dir, f"{cfg.run_name}_step{step}.pt")
                save_checkpoint(p, step, cfg, model, encoder, transport, opt, ema)
                print(f"  saved {p}")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min. log: {log_path}")
    return log_path
