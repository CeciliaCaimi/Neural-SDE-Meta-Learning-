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
import time

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

# The source-dependence panel (--source-controls). Each row keeps the relation descriptor
# and destroys task identity a different way, so the gap between "transport" and these
# three is what knowing the source task is worth. All three skip refinement, matching the
# "transport" row, so that no target image enters any of them.
CONTROLS = [
    ("mean z_S",      "transport_no_refine", "the mean source coordinate of this transformation"),
    ("shuffled z_S",  "transport_no_refine", "another class's source coordinate, same transformation"),
    ("relation only", "relation_only",       "transport from the relation alone, z_S zeroed"),
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
    ap.add_argument("--m-source", type=int, default=None,
                    help="how many source images the encoder sees (M_S). Passing it also "
                         "switches the source support to nested prefixes so that a sweep "
                         "varies set size alone; the upper bound is the source support "
                         "pool, 300 by default")
    ap.add_argument("--json-out", default=None,
                    help="write every per-episode value to this file. Summaries are cheap "
                         "to recompute from it and expensive to recompute from nothing: "
                         "without it, asking a different question of the same run means "
                         "generating every sample again.")
    ap.add_argument("--progress", type=int, default=0, metavar="N",
                    help="report progress to stderr every N episodes. The tables are only "
                         "printed once the whole sweep is done, so without this a long run "
                         "is indistinguishable from a stalled one. Goes to stderr so the "
                         "results file stays clean; redirect the two streams separately.")
    ap.add_argument("--source-controls", action="store_true",
                    help="add the mean, within-relation shuffled and relation-only source "
                         "conditions, which together say what the source task is worth")
    ap.add_argument("--grid-out", default=None,
                    help="write a paired grid of what each strategy generates, at the "
                         "smallest K_T, sharing one initial noise down each column")
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
    print(f"sampling   DDIM eta=0, {a.ddim_steps} steps, {a.n_samples} samples per cell")
    m_source = a.m_source or cfg.episodes.enc_source_images
    print(f"source     M_S={m_source}, sets {'nested (prefix)' if a.m_source else 'drawn independently'}\n")

    raw = load_cifar100()
    loader = DomainShiftLoader(raw, split, a.split, device=dev,
                               enc_source_images=m_source,
                               query_batch=128, seed=a.seed,
                               nested_source=a.m_source is not None)
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

    strategies = list(STRATEGIES) + (CONTROLS if a.source_controls else [])

    # ---- the source-dependence controls need coordinates from other episodes ---------
    z_s_all, partner, mean_by_cor = [], {}, {}
    if a.source_controls:
        with torch.no_grad():
            z_s_all = [enc(e["batch"].src_support) for e in episodes]
        by_cor: dict[str, list[int]] = {}
        for i, e in enumerate(episodes):
            by_cor.setdefault(e["cor"], []).append(i)
        mean_by_cor = {c: torch.stack([z_s_all[i] for i in idx]).mean(0)
                       for c, idx in by_cor.items()}
        for c, idx in by_cor.items():
            for pos, i in enumerate(idx):
                # the nearest episode of the same transformation whose class differs: the
                # shuffle must hold the relation fixed and destroy only task identity
                cand = [j for j in idx[pos + 1:] + idx[:pos]
                        if episodes[j]["fid"] != episodes[i]["fid"]]
                if not cand:
                    raise SystemExit(
                        f"transformation '{c}' covers only one class here, so no "
                        f"within-relation shuffled source exists; raise --n-classes")
                partner[i] = cand[0]
        print(f"source controls on: mean over {len(cors)} transformations, "
              f"shuffle within transformation, relation-only\n")

    # the scale the three statistics are compared on
    f_sd = torch.cat([transform_signature(e["batch"].tgt_query) for e in episodes]).std(0)

    res = {name: {k: {"sig": [], "sw": [], "mmd": []} for k in a.k_shots}
           for name, _, _ in strategies}
    k_grid = min(a.k_shots)
    grid_rows = {name: [] for name, _, _ in strategies}
    grid_real = []
    t_start = time.time()
    for i, e in enumerate(episodes):
        if a.progress and i and i % a.progress == 0:
            rate = (time.time() - t_start) / i
            print(f"  {i}/{len(episodes)} episodes, {rate:.1f}s each, "
                  f"~{rate * (len(episodes) - i) / 60:.0f} min left",
                  file=sys.stderr, flush=True)
        b, k = e["batch"], e["k"]
        real = b.tgt_query
        real_sig = transform_signature(real).mean(0)
        seed = a.seed + 100 * i                       # shared across strategies: paired
        want_grid = a.grid_out and k == k_grid and len(grid_real) < 9
        if want_grid:
            grid_real.append(real[:3].cpu())
        for name, strat, _ in strategies:
            oracle_data = b.tgt_query if strat == "oracle" else None
            ov = None
            if name == "mean z_S":
                ov = mean_by_cor[e["cor"]]
            elif name == "shuffled z_S":
                ov = z_s_all[partner[i]]
            # one refinement noise stream per episode, rewound for every strategy, so the
            # (t, eps) a strategy refines against cannot differ from its competitor's
            rgen = torch.Generator(device=dev).manual_seed(seed + 1)
            st = adapt(strat, model, enc, tr, b, budget,
                       cfg.diffusion.loss_weighting, oracle_data=oracle_data,
                       z_s_override=ov, generator=rgen)
            x = generate(model, st.z, a.n_samples, seed, a.ddim_steps, dev)
            if want_grid:
                grid_rows[name].append(x[:3].cpu())
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
        for name, _, _ in strategies:
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
    if a.source_controls:
        # what the source task is worth, against each way of destroying it while keeping
        # the relation. A1 is decided on these three rows.
        pairs += [
            ("transport vs mean z_S", "mean z_S", "transport"),
            ("transport vs shuffled z_S", "shuffled z_S", "transport"),
            ("transport vs relation only", "relation only", "transport"),
        ]
    # Both metrics, because they can disagree: the transformation statistic measures the
    # axis the coordinate was shown to control, the sliced Wasserstein distance is the
    # headline distributional metric, and a conclusion that holds on only one of them is
    # a conclusion about the instrument.
    for metric, what in (("sig", "transformation-statistic space"),
                         ("sw", "sliced Wasserstein, pixel space")):
        hdr = f"\n  {'comparison, ' + what:<34}" + "".join(
            f"{'K_T=' + str(k):>18}" for k in a.k_shots)
        print(hdr)
        print("  " + "-" * (len(hdr) - 3))
        for label, worse, better in pairs:
            row = f"  {label:<34}"
            for k in a.k_shots:
                d = [x - y for x, y in zip(res[worse][k][metric], res[better][k][metric])]
                mu, h = ci95(d)
                mark = "*" if abs(mu) > h else " "
                row += f"{mu:>+11.4f} +-{h:<5.4f}{mark}"
            print(row)
    print("\n  * = the 95% interval excludes zero")

    if a.json_out:
        import json
        with open(a.json_out, "w", encoding="utf-8") as fo:
            json.dump({"checkpoint": os.path.basename(a.ckpt), "step": step,
                       "split": a.split, "k_shots": a.k_shots,
                       "n_samples": a.n_samples, "ddim_steps": a.ddim_steps,
                       "m_source": m_source, "nested_source": a.m_source is not None,
                       "episodes": [{"fid": e["fid"], "cor": e["cor"], "k": e["k"]}
                                    for e in episodes],
                       "per_episode": {name: {str(k): res[name][k] for k in a.k_shots}
                                       for name, _, _ in strategies}}, fo)
        print(f"per-episode values written to {a.json_out}")

    if a.grid_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [n for n, _, _ in strategies] + ["real target"]
        rows = [torch.cat(grid_rows[n], dim=0) for n, _, _ in strategies]
        rows.append(torch.cat(grid_real, dim=0))
        n_show = min(8, rows[0].shape[0])
        fig, axes = plt.subplots(len(rows), n_show,
                                 figsize=(n_show * 1.15, len(rows) * 1.28))
        for r, (lab, row) in enumerate(zip(labels, rows)):
            for c in range(n_show):
                ax = axes[r, c]
                ax.imshow(((row[c].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
        fig.suptitle(f"one shared initial noise down each column, K_T = {k_grid}; "
                     "the bottom row is real target data, not generated", fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.grid_out) or ".", exist_ok=True)
        fig.savefig(a.grid_out, dpi=130)
        print(f"\ngrid written to {a.grid_out}")


if __name__ == "__main__":
    main()
