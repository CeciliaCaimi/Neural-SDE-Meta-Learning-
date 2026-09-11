"""Do the deployment strategies differ, when compared where the effect is visible?

    python scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt \
        --domainshift-path artifacts/cifar100_domainshift_c3.json --split val

Two questions that the denoising loss has already been asked and answered "no" to, and
that are worth asking again now that the loss is known to compress this effect by a large
factor.

1. **Does transport beat the baselines?** The deployment table puts every strategy on
   0.0887 -- transport, target-only, refinement and the oracle upper bound alike -- with
   only source-reuse separating at small K_T. If the loss cannot resolve a difference that
   is plain in the samples, it cannot be trusted to say the strategies are equal either.

2. **Does refinement do anything?** It is currently worth -0.0000 against not refining.
   But refinement optimises the denoising loss, and the coordinate-loss profile shows that
   objective is nearly flat around the right coordinate: the whole excursion from the
   shared mean to the episode's own coordinate is 0.00135. Refinement may be searching a
   surface too flat to descend, which is a different failure from being useless.

The measurement is the distance between the generated set and the real target set, in the
three-statistic space that needs no trained instrument. Everything is paired: within one
episode every strategy generates from the identical initial noise.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from adaptation.budget import AdaptBudget                                 # noqa: E402
from adaptation.coordinate import adapt                                   # noqa: E402
from diffusion.sampler import ddim_sample, make_eps_fn                    # noqa: E402
from domains.cifar100 import load_cifar100                                # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift      # noqa: E402
from evaluation.instruments import transform_signature                    # noqa: E402
from evaluation.metrics_analytic import energy_mmd, sliced_wasserstein    # noqa: E402
from posthoc_controls import ci95, load_checkpoint                        # noqa: E402

# name, strategy in adaptation/coordinate.py, what it is
STRATEGIES = [
    ("no adaptation",      "zero",                "z = 0, nothing fitted"),
    ("reuse z_S",          "source_reuse",        "the source coordinate, no transport"),
    ("transport",          "transport_no_refine", "transport only, no target image used"),
    ("transport + refine", "transport",           "the method: transport, then K_T images"),
    ("target only",        "target_only",         "encode the K_T images, no source"),
    ("oracle",             "oracle",              "refined on abundant target data"),
]


@torch.no_grad()
def generate(model, z, n: int, seed: int, steps: int, dev) -> torch.Tensor:
    gen = torch.Generator(device=dev).manual_seed(seed)
    return ddim_sample(model.schedule, make_eps_fn(model, z), (n, 3, 32, 32), dev,
                       n_steps=steps, eta=0.0, generator=gen, clip_x0=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--domainshift-path", default=None)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4321)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, enc, tr, cfg, step, used_ema = load_checkpoint(a.ckpt, dev)

    ds_path = a.domainshift_path or cfg.episodes.domainshift_path
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(_ROOT, ds_path)
    split = load_domainshift(ds_path)
    cors = tuple(split.config.corruptions)
    if len(cors) != (cfg.model.n_relations or 1):
        raise SystemExit("split/checkpoint mismatch")

    budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0,
                         noise_batch=cfg.adapt.noise_batch)
    print(f"checkpoint {os.path.basename(a.ckpt)}  step {step}  "
          f"weights {'EMA' if used_ema else 'raw'}")
    print(f"refinement J={budget.steps}, eta_z={budget.lr}, beta_0={budget.beta0}")
    print(f"sampling   DDIM eta=0, {a.ddim_steps} steps, {a.n_samples} samples per cell\n")

    raw = load_cifar100()
    loader = DomainShiftLoader(raw, split, a.split, device=dev,
                               enc_source_images=cfg.episodes.enc_source_images,
                               query_batch=128, seed=a.seed)
    all_fids = split.fine_ids(a.split)
    fids = all_fids[:a.n_classes]

    # ---- one episode per (class, transformation, K_T) -------------------------------
    episodes = []
    for fid in fids:
        loader.fids = [fid]
        for c in cors:
            for k in a.k_shots:
                episodes.append(dict(fid=fid, cor=c, k=k,
                                     batch=loader.sample(k_shot=k, corruption=c)))
    loader.fids = all_fids
    print(f"{len(episodes)} episodes: {len(fids)} classes x {len(cors)} transformations "
          f"x {len(a.k_shots)} values of K_T\n")

    # the scale the three statistics are compared on
    f_sd = torch.cat([transform_signature(e["batch"].tgt_query) for e in episodes]).std(0)

    res = {name: {k: {"sig": [], "sw": [], "mmd": []} for k in a.k_shots}
           for name, _, _ in STRATEGIES}
    for i, e in enumerate(episodes):
        b, k = e["batch"], e["k"]
        real = b.tgt_query
        real_sig = transform_signature(real).mean(0)
        seed = a.seed + 100 * i                       # shared across strategies: paired
        for name, strat, _ in STRATEGIES:
            oracle_data = b.tgt_query if strat == "oracle" else None
            st = adapt(strat, model, enc, tr, b, budget,
                       cfg.diffusion.loss_weighting, oracle_data=oracle_data)
            x = generate(model, st.z, a.n_samples, seed, a.ddim_steps, dev)
            sig = transform_signature(x).mean(0)
            res[name][k]["sig"].append(float((((sig - real_sig) / f_sd) ** 2).sum().sqrt()))
            fa, fb = x.flatten(1), real.flatten(1)
            g = torch.Generator(device=dev).manual_seed(0)
            res[name][k]["sw"].append(sliced_wasserstein(fa, fb, n_proj=256, generator=g))
            res[name][k]["mmd"].append(energy_mmd(fa[:64], fb[:64]))

    def table(metric: str, title: str, note: str) -> None:
        print(f"\n{title}")
        print(f"  {note}")
        hdr = f"\n  {'strategy':<20}" + "".join(f"{'K_T=' + str(k):>16}" for k in a.k_shots)
        print(hdr)
        print("  " + "-" * (len(hdr) - 3))
        for name, _, _ in STRATEGIES:
            row = f"  {name:<20}"
            for k in a.k_shots:
                mu, h = ci95(res[name][k][metric])
                row += f"{mu:>10.3f} +-{h:<4.3f}"
            print(row)

    table("sig", "distance from the real target domain, in transformation-statistic space",
          "lower is better; this is the axis the coordinate was shown to control")
    table("sw", "sliced Wasserstein distance to the real target set, pixel space",
          "lower is better")

    # ---- the two questions, answered as paired differences --------------------------
    print("\n\nthe two questions, as paired differences over episodes (positive = the "
          "first one is worse)")
    pairs = [
        ("transport vs reuse z_S", "reuse z_S", "transport"),
        ("transport vs target only", "target only", "transport"),
        ("refinement: off vs on", "transport", "transport + refine"),
        ("transport + refine vs oracle", "transport + refine", "oracle"),
    ]
    hdr = f"\n  {'comparison':<30}" + "".join(f"{'K_T=' + str(k):>18}" for k in a.k_shots)
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for label, worse, better in pairs:
        row = f"  {label:<30}"
        for k in a.k_shots:
            d = [x - y for x, y in zip(res[worse][k]["sig"], res[better][k]["sig"])]
            mu, h = ci95(d)
            mark = "*" if abs(mu) > h else " "
            row += f"{mu:>+11.3f} +-{h:<4.3f}{mark}"
        print(row)
    print("\n  * = the 95% interval excludes zero")


if __name__ == "__main__":
    main()
