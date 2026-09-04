"""Adaptation-strategy comparison for the CIFAR domain shift.

    python scripts/cifar_table.py checkpoints/cifar_ds3_step50000.pt

Key difference from the GMM side: images have **no analytic ground truth**, so the relative
score-field error cannot be computed. The metric is instead the **denoising loss** on held-out
target query images (lower is better): a proxy for the ELBO, on the training objective scale.

Paired statistics: within one (class, corruption, K_T) triple every strategy sees the same
query batch and the same noising; differences are taken per episode, cancelling task difficulty.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from adaptation.budget import AdaptBudget                       # noqa: E402
from adaptation.coordinate import adapt                         # noqa: E402
from config.base_config import BaseConfig                       # noqa: E402
from diffusion.forward import q_sample                          # noqa: E402
from domains.cifar100 import load_cifar100                      # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift   # noqa: E402
from training.loop import build                                 # noqa: E402

STRATEGIES = [
    ("no_adapt",        "zero",                 "no adaptation (z=0)"),
    ("target_only",     "target_only",          "target support only"),
    ("source_reuse",    "source_reuse",         "reuse z_S directly"),
    ("transport_only",  "transport_no_refine",  "transport, no refine"),
    ("transport",       "transport",            "transport + refine (ours)"),
    ("oracle",          "oracle",               "oracle (fit on target)"),
]


@torch.no_grad()
def apply_ema(modules, shadow) -> None:
    """EMA shadow weights are a list of dicts in module order (see EMA.copy_to in training/loop.py)."""
    for m, sh in zip(modules, shadow):
        sd = m.state_dict()
        for k, v in sh.items():
            sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def eval_loss(model, z, tgt_query, sched, n_noise: int, gen) -> float:
    """Denoising loss on held-out target query images. (t, eps) is fixed so strategies pair up."""
    tot = 0.0
    for _ in range(n_noise):
        nb = q_sample(sched, tgt_query, generator=gen)
        eps_hat = model.eps_hat(nb.x_t, nb.t, z.unsqueeze(0).expand(len(tgt_query), -1))
        tot += float((eps_hat - nb.eps).pow(2).mean())
    return tot / n_noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--n-noise", type=int, default=8, help="how many (t, eps) draws to average")
    a = ap.parse_args()

    dev = torch.device("cuda")
    cfg = BaseConfig()
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg.episodes.scheme = "domainshift"

    # Rebuild the model from the **checkpoint's own** config, never from the current
    # defaults: k, backbone size, pooling and relation count all change the shapes, and
    # a config drift since training would otherwise fail obscurely or, worse, load a
    # mismatched model silently.
    ck_model = sd["config"]["model"]
    cfg.model.k = ck_model["k"]
    cfg.model.backbone = ck_model["backbone"]
    cfg.model.backbone_kwargs = dict(ck_model["backbone_kwargs"])
    cfg.model.encoder_pooling = ck_model.get("encoder_pooling", "mean")
    cfg.model.n_relations = ck_model["n_relations"]

    split = load_domainshift(os.path.join(_ROOT, cfg.episodes.domainshift_path))
    # The evaluation data must come from the same task family the checkpoint was trained
    # on. n_relations records how many corruptions that was, so a mismatch means the split
    # file has been rebuilt with different settings and the numbers would be meaningless.
    n_cor = len(split.config.corruptions)
    trained_cor = cfg.model.n_relations or 1
    if n_cor != trained_cor:
        raise SystemExit(
            f"split/checkpoint mismatch: the split file has {n_cor} corruption(s) "
            f"{split.config.corruptions}, but this checkpoint was trained with "
            f"{trained_cor}. Rebuild the split with the training settings before evaluating."
        )

    model, enc, tr = build(cfg, dev)
    model.load_state_dict(sd["model"])
    enc.load_state_dict(sd["encoder"]); tr.load_state_dict(sd["transport"])
    used_ema = "ema" in sd and sd["ema"] is not None
    if used_ema:
        apply_ema([model, enc, tr], sd["ema"])
    for m in (model, enc, tr):
        m.eval()
    print(f"checkpoint {os.path.basename(a.ckpt)}  step {sd.get('step','?')}  "
          f"weights {'EMA' if used_ema else 'raw'}  corruptions {split.config.corruptions}")

    raw = load_cifar100()
    budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0)

    # results[k][strategy] = per-episode losses
    results = {k: {name: [] for name, _, _ in STRATEGIES} for k in a.k_shots}
    for k in a.k_shots:
        ld = DomainShiftLoader(raw, split, "test", device=dev,
                               enc_source_images=cfg.episodes.enc_source_images,
                               query_batch=cfg.episodes.query_batch, seed=1234)
        for ep in range(a.n_episodes):
            b = ld.sample(k_shot=k)
            gen = torch.Generator(device=dev).manual_seed(9000 + ep)  # shared across strategies
            for name, strat, _ in STRATEGIES:
                oracle_data = b.tgt_query if strat == "oracle" else None
                st = adapt(strat, model, enc, tr, b, budget,
                           cfg.diffusion.loss_weighting, oracle_data=oracle_data)
                gen.manual_seed(9000 + ep)          # every strategy sees the same (t, eps)
                results[k][name].append(
                    eval_loss(model, st.z, b.tgt_query, model.schedule, a.n_noise, gen))

    ours = "transport"
    print(f"\n{'strategy':<26}" + "".join(f"{'K_T='+str(k):>14}" for k in a.k_shots))
    print("-" * (26 + 14 * len(a.k_shots)))
    for name, _, label in STRATEGIES:
        row = f"{label:<26}"
        for k in a.k_shots:
            row += f"{sum(results[k][name])/len(results[k][name]):>14.4f}"
        print(row)

    print(f"\npaired advantage of this method over each baseline (positive = better, +- 95% CI)")
    print(f"{'baseline':<26}" + "".join(f"{'K_T='+str(k):>20}" for k in a.k_shots))
    print("-" * (26 + 20 * len(a.k_shots)))
    for name, _, label in STRATEGIES:
        if name == ours:
            continue
        row = f"{label:<26}"
        for k in a.k_shots:
            d = torch.tensor(results[k][name]) - torch.tensor(results[k][ours])
            h = 1.96 * float(d.std()) / math.sqrt(len(d))
            row += f"{float(d.mean()):>+13.4f} ±{h:.4f}"
        print(row)
    print(f"\n({a.n_episodes} held-out episodes x {a.n_noise} noise draws, paired per episode)")


if __name__ == "__main__":
    main()
