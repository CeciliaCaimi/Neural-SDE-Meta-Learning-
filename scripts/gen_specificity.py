"""Does the task coordinate change what the model *generates*?

    python scripts/gen_specificity.py checkpoints/cifar_ds3_step50000.pt \
        --domainshift-path artifacts/cifar100_domainshift_c3.json --split val

Everything measured so far is denoising loss on real images pushed forward through the
noising process. That is not where samples come from. Generation runs the reverse process
from pure noise, visits every timestep, and compounds its own error for fifty steps; a
coordinate could move it without moving a loss averaged over the forward process, or move
the loss without moving it.

Four coordinates are compared from **one shared column of initial noise**, so the only
thing that differs between the four images in a row is the coordinate:

    z_own       this episode's transported coordinate
    z_mean      one coordinate shared by every episode -- task-specific content removed
    z_shuffled  another episode's
    z = 0       no coordinate at all

Two measurements, chosen because the controls already told us what to expect. The
coordinate was measured to carry transformation identity (+0.00133) and essentially no
class identity (+0.00002), so the prediction under test is that generation separates by
transformation and not by class. That is falsifiable in both directions.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from diffusion.sampler import ddim_sample, make_eps_fn                    # noqa: E402
from domains.cifar100 import load_cifar100                                # noqa: E402
from domains.corruptions import CORRUPTIONS                               # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift      # noqa: E402
from evaluation.instruments import (                                      # noqa: E402
    train_class_instrument, transform_signature,
)
from evaluation.metrics_analytic import energy_mmd, sliced_wasserstein    # noqa: E402
from posthoc_controls import ci95, episode_coordinates, load_checkpoint   # noqa: E402

COORDS = ("own", "mean", "shuffled", "zero")


@torch.no_grad()
def generate(model, z, n: int, seed: int, steps: int, dev) -> torch.Tensor:
    """One batch of samples. The generator is seeded per episode, not per coordinate, so
    every coordinate starts from the identical x_T and the comparison is paired."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    return ddim_sample(model.schedule, make_eps_fn(model, z), (n, 3, 32, 32), dev,
                       n_steps=steps, eta=0.0, generator=gen, clip_x0=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--domainshift-path", default=None)
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n-classes", type=int, default=8, help="val classes to cover")
    ap.add_argument("--n-samples", type=int, default=128, help="samples per (episode, coordinate)")
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--grid-out", default=None, help="write a paired visual grid here")
    ap.add_argument("--skip-classifier", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, enc, tr, cfg, step, used_ema = load_checkpoint(a.ckpt, dev)

    ds_path = a.domainshift_path or cfg.episodes.domainshift_path
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(_ROOT, ds_path)
    split = load_domainshift(ds_path)
    cors = tuple(split.config.corruptions)
    if len(cors) != (cfg.model.n_relations or 1):
        raise SystemExit(f"split/checkpoint mismatch: {len(cors)} vs "
                         f"{cfg.model.n_relations or 1} relations")

    print(f"checkpoint {os.path.basename(a.ckpt)}  step {step}  "
          f"weights {'EMA' if used_ema else 'raw'}")
    print(f"split      {os.path.basename(ds_path)}  relations {cors}")
    print(f"sampling   DDIM eta=0, {a.ddim_steps} steps, {a.n_samples} samples per cell, "
          f"shared initial noise across the four coordinates\n")

    raw = load_cifar100()
    loader = DomainShiftLoader(raw, split, a.split, device=dev,
                               enc_source_images=cfg.episodes.enc_source_images,
                               query_batch=128, seed=a.seed)
    all_fids = split.fine_ids(a.split)
    fids = all_fids[:a.n_classes]

    instrument, meta = (None, None)
    if not a.skip_classifier:
        instrument, meta = train_class_instrument(raw, split, a.split, dev, seed=a.seed)
        print("")

    # ---- episodes: every (class, transformation) pair over the chosen classes --------
    episodes = []
    for fid in fids:
        loader.fids = [fid]                       # force the class; the loader picks at random otherwise
        refs = {}
        for c in cors:
            b = loader.sample(k_shot=1, corruption=c)
            refs[c] = b.tgt_query
            episodes.append(dict(fid=fid, cor=c, batch=b, ref=b.tgt_query))
        for e in episodes[-len(cors):]:
            e["refs_all"] = refs                  # the same class under each transformation
    loader.fids = all_fids

    for e in episodes:
        h = e["batch"].tgt_query.shape[0] // 2
        _, z_tld, _ = episode_coordinates(enc, tr, e["batch"], e["batch"].tgt_query[:h])
        e["z"] = z_tld

    z_bar = torch.stack([e["z"] for e in episodes]).mean(dim=0)
    print(f"{len(episodes)} episodes: {len(fids)} classes x {len(cors)} transformations\n")

    # ---- the scale on which the three statistics are compared ------------------------
    ref_feats = torch.cat([transform_signature(e["ref"]) for e in episodes])
    f_mu, f_sd = ref_feats.mean(0), ref_feats.std(0)

    # ---- generate and measure --------------------------------------------------------
    res = {c: {"trans_hit": [], "class_hit": [], "sw": [], "mmd": [], "sig": []} for c in COORDS}
    grid_rows = {c: [] for c in COORDS}
    for i, e in enumerate(episodes):
        j = (i + len(cors) + 1) % len(episodes)            # another episode, usually another transformation
        zs = {"own": e["z"], "mean": z_bar, "shuffled": episodes[j]["z"],
              "zero": torch.zeros_like(e["z"])}
        seed = a.seed + 100 * i
        for c in COORDS:
            x = generate(model, zs[c], a.n_samples, seed, a.ddim_steps, dev)

            sig = transform_signature(x).mean(0)
            res[c]["sig"].append(sig)
            # which transformation does this set look like? nearest real centroid of the
            # same class, in z-scored statistic space. Chance is 1/len(cors).
            d = [float((((sig - transform_signature(e["refs_all"][cc]).mean(0)) / f_sd) ** 2).sum())
                 for cc in cors]
            res[c]["trans_hit"].append(1.0 if cors[int(np.argmin(d))] == e["cor"] else 0.0)

            if instrument is not None:
                p = torch.softmax(instrument(x), dim=1).mean(0)
                res[c]["class_hit"].append(
                    1.0 if meta["fids"][int(p.argmax())] == e["fid"] else 0.0)

            fa, fb = x.flatten(1), e["ref"].flatten(1)
            g = torch.Generator(device=dev).manual_seed(0)
            res[c]["sw"].append(sliced_wasserstein(fa, fb, n_proj=256, generator=g))
            res[c]["mmd"].append(energy_mmd(fa[:64], fb[:64]))

            if a.grid_out and i < 3:
                grid_rows[c].append(x[:8].cpu())

    # ---- report ----------------------------------------------------------------------
    n_ep = len(episodes)
    chance_t = 1.0 / len(cors)
    print("does generation carry the task? every figure is over "
          f"{n_ep} episodes, paired on initial noise\n")
    print(f"  {'coordinate':<12}{'transformation':>16}{'95% CI':>10}"
          f"{'class':>10}{'95% CI':>10}{'SW2':>9}{'energy MMD':>12}")
    print("  " + "-" * 79)
    for c in COORDS:
        tm, th = ci95(res[c]["trans_hit"])
        cm, ch = ci95(res[c]["class_hit"]) if res[c]["class_hit"] else (float("nan"), float("nan"))
        sw = sum(res[c]["sw"]) / n_ep
        mmd = sum(res[c]["mmd"]) / n_ep
        print(f"  {c:<12}{100*tm:>15.1f}%{100*th:>9.1f}{100*cm:>9.1f}%{100*ch:>9.1f}"
              f"{sw:>9.3f}{mmd:>12.4f}")
    print(f"\n  chance: transformation {100*chance_t:.0f}%"
          + (f", class {100/meta['n_fine']:.0f}%" if meta else ""))
    if meta:
        print(f"  the instrument's own ceiling on real held-out images: "
              f"{100*np.mean(list(meta['acc_fine'].values())):.1f}% fine, averaged over domains")

    # The aggregate hides the question. What matters is whether sets generated *for a
    # blur episode* differ from sets generated *for a noise episode* -- that is the
    # coordinate carrying transformation identity into the samples, or failing to.
    print("\nhigh-frequency energy of the generated sets, grouped by the episode's "
          "true transformation")
    print("  a coordinate that carries the transformation should spread these rows apart")
    hdr = f"  {'coordinate':<12}" + "".join(f"{c[:9]:>12}" for c in cors) + f"{'spread':>10}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in COORDS:
        by = {cc: [] for cc in cors}
        for e, s in zip(episodes, res[c]["sig"]):
            by[e["cor"]].append(float(s[0]))
        means = [sum(by[cc]) / max(1, len(by[cc])) for cc in cors]
        print(f"  {c:<12}" + "".join(f"{m:>12.4f}" for m in means)
              + f"{max(means) - min(means):>10.4f}")
    real = []
    for cc in cors:
        s = torch.cat([transform_signature(e["refs_all"][cc]) for e in episodes]).mean(0)
        real.append(float(s[0]))
    print(f"  {'real target':<12}" + "".join(f"{m:>12.4f}" for m in real)
          + f"{max(real) - min(real):>10.4f}")

    print("\nfull statistic triple, averaged over all episodes")
    print(f"  {'coordinate':<12}{'HF energy':>12}{'std':>10}{'residual':>11}")
    print("  " + "-" * 45)
    for c in COORDS:
        s = torch.stack(res[c]["sig"]).mean(0)
        print(f"  {c:<12}{s[0]:>12.4f}{s[1]:>10.4f}{s[2]:>11.4f}")
    for cc in cors:
        s = torch.cat([transform_signature(e["refs_all"][cc]) for e in episodes]).mean(0)
        print(f"  {'real ' + cc[:7]:<12}{s[0]:>12.4f}{s[1]:>10.4f}{s[2]:>11.4f}")

    if a.grid_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [r for c in COORDS for r in [torch.cat(grid_rows[c], dim=0)]]
        n_show = min(8, rows[0].shape[0])
        fig, axes = plt.subplots(len(COORDS), n_show, figsize=(n_show * 1.15, len(COORDS) * 1.3))
        for r, (c, row) in enumerate(zip(COORDS, rows)):
            for k in range(n_show):
                ax = axes[r, k]
                ax.imshow(((row[k].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(c, fontsize=9)
        fig.suptitle("same initial noise down each column; only the coordinate differs", fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.grid_out) or ".", exist_ok=True)
        fig.savefig(a.grid_out, dpi=130)
        print(f"\ngrid written to {a.grid_out}")


if __name__ == "__main__":
    main()
