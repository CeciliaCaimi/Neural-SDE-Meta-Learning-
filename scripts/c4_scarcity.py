"""C4: the demographic scarcity sweep on Fitzpatrick17k.

    python scripts/c4_scarcity.py checkpoints/c_fitz_step50000.pt --split test

Abundant source-population evidence, scarce target-population evidence, and the question of
whether knowing the source task plus the learned relation beats every way of not knowing it.
The arms are the plan's, one per line of C4.

**Read this before reading the table.** The unit of replication is the *task*, and with one
relation a condition is a task, so the test side gives **three**. Stage A had sixty episodes
per K_T. A 95 % interval over three tasks is not a substitute for that, so this script reports
each task on its own row and the across-task interval second, and any claim it supports is
directional. `handoff/STAGE_C_DATA.md` section 6 explains why more images, more samples and a
denser K_T grid all leave the three untouched, and names the only two things that move it.

The source support is resampled ``--repeats`` times per (task, K_T) cell. That averages out
which sixteen source photographs the encoder happened to see -- which is real noise and worth
removing -- but it creates no new tasks, and the across-task interval is computed over tasks,
never over resamples.

The headline instrument is fixed, not trained: ``phototype_signature`` reads the individual
typology angle, which is monotone across all six Fitzpatrick levels on the real data
(45.6, 30.6, 21.4, 14.1, 6.1, -14.9) with nothing learned. The trained predictors are
secondary and print their own ceilings beside every verdict.
"""
from __future__ import annotations

import argparse
import json
import math
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
from domains.fitzpatrick import load_fitzpatrick                          # noqa: E402
from episodes.fitzpatrick import FitzLoader, load_fitz_split              # noqa: E402
from evaluation.fitz_instruments import (                                 # noqa: E402
    condition_share, phenotype_target_share, phototype_signature,
    train_condition_instrument, train_phenotype_instrument,
)
from evaluation.metrics_analytic import energy_mmd, kid, sliced_wasserstein  # noqa: E402
from posthoc_controls import ci95, load_checkpoint                        # noqa: E402

# The six arms of C4, in the plan's order. "population only" is what the plan calls
# population / relation only: with a single relation it is Delta_gamma(0), one constant, and
# it stops being distinguishable from a mean-source control -- which is itself the finding
# that this task family has only one relation to learn.
ARMS = [
    ("no adaptation",   "zero",                 "z = 0, nothing fitted"),
    ("target only",     "target_only",          "encode the K_T target images, no source"),
    ("reuse z_S",       "source_reuse",         "the source coordinate, no transport"),
    ("population only", "relation_only",        "the relation alone, z_S zeroed"),
    ("wrong source",    "transport_no_refine",  "another condition's source coordinate"),
    ("transport",       "transport_no_refine",  "the method: correct source, then transport"),
    ("transport+refine", "transport",           "transport, then the K_T target images"),
    ("oracle",          "oracle",               "refined on the abundant held-out target set"),
]


@torch.no_grad()
def generate(model, z, n: int, seed: int, steps: int, dev, size: int) -> torch.Tensor:
    gen = torch.Generator(device=dev).manual_seed(seed)
    return ddim_sample(model.schedule, make_eps_fn(model, z), (n, 3, size, size), dev,
                       n_steps=steps, eta=0.0, generator=gen, clip_x0=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--fitz-path", default="artifacts/fitzpatrick_split.json")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    # The plan's C4 grid is {1, 2, 5, 10, 20}. The Fitzpatrick run used {1,2,3,5,8,12,20},
    # which omits K_T = 10 -- and 10 is the point the plan's headline form needs ("the method
    # at K_T = 2 matches target-only at K_T = 10"). The default now follows the plan.
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--repeats", type=int, default=8,
                    help="source-support resamples per (task, K_T). Averages out which "
                         "source photographs the encoder saw; creates no new tasks.")
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--no-instruments", action="store_true",
                    help="skip the two trained predictors; the fixed ITA statistic and the "
                         "distributional metrics still report")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--grid-out", default=None)
    ap.add_argument("--progress", type=int, default=0, metavar="N")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, enc, tr, cfg, step, used_ema = load_checkpoint(a.ckpt, dev)
    size = model.image_size

    path = a.fitz_path if os.path.isabs(a.fitz_path) else os.path.join(_ROOT, a.fitz_path)
    split = load_fitz_split(path)
    if max(a.k_shots) > split.max_k:
        raise SystemExit(f"the split reserves {split.max_k} target support images; "
                         f"K_T={max(a.k_shots)} asked for")

    print(f"checkpoint {os.path.basename(a.ckpt)}  step {step}  "
          f"weights {'EMA' if used_ema else 'raw'}")
    print(f"arm        {cfg.model.score_model} / {cfg.model.backbone}  k={cfg.model.k}  "
          f"transport {cfg.model.transport_kind}")
    print(f"split      {os.path.basename(path)}  checksum {split.checksum[:12]}...  "
          f"{split.relation['source']} -> {split.relation['target']}")
    print(f"sampling   DDIM eta=0, {a.ddim_steps} steps, {a.n_samples} samples per cell, "
          f"{size}x{size}")

    # every image any stream will ask for, so decoding touches nothing else
    raw = load_fitzpatrick(image_size=size, hashes=split.all_hashes())
    loader = FitzLoader(raw, split, a.split, device=dev, query_batch=512,
                        k_shots=tuple(a.k_shots), seed=a.seed)
    conditions = split.names(a.split)
    print(f"tasks      {len(conditions)} held-out conditions: {', '.join(conditions)}")
    for c in conditions:
        print(f"             {c:<44} {len(split.conditions[c].tgt_query)} held-out "
              f"target images")

    budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0,
                         noise_batch=cfg.adapt.noise_batch)

    # ---- the instruments -----------------------------------------------------------
    ph_rep = cd_rep = None
    if not a.no_instruments:
        print()
        full = load_fitzpatrick(image_size=size, verbose=False)
        excl = {h for c in split.conditions.values() for h in c.tgt_query}
        test_h = {h for n in split.names(a.split)
                  for h in split.conditions[n].all_hashes()}
        ph_rep = train_phenotype_instrument(full, exclude=excl, test_hashes=test_h,
                                           device=dev, seed=a.seed)
        cd_rep = train_condition_instrument(full, exclude=excl, device=dev,
                                           must_include=tuple(conditions), seed=a.seed)
        del full

    # ---- the real reference sets, and the scale the fixed statistic is compared on ---
    real = {c: loader.sample(k_shot=1, condition=c).tgt_query for c in conditions}
    all_real_sig = torch.cat([phototype_signature(v) for v in real.values()])
    sig_sd = all_real_sig.std(0).clamp_min(1e-6)
    print("\nthe fixed population statistic on the real held-out target sets")
    print(f"  {'condition':<44}{'ITA':>9}{'L*':>9}{'b*':>9}{'a*':>9}   n")
    for c in conditions:
        s = phototype_signature(real[c]).mean(0)
        print(f"  {c:<44}" + "".join(f"{float(v):>9.2f}" for v in s)
              + f"   {real[c].shape[0]}")
    src_ref = {c: loader.sample(k_shot=1, condition=c).src_support for c in conditions}
    print("  -- for contrast, the source-population support of each task --")
    for c in conditions:
        s = phototype_signature(src_ref[c]).mean(0)
        print(f"  {c:<44}" + "".join(f"{float(v):>9.2f}" for v in s))

    # ---- the sweep -----------------------------------------------------------------
    # res[arm][k][condition] = list over repeats
    res = {name: {k: {c: {m: [] for m in ("sig", "sw", "mmd", "ita", "ph", "cls")}
                      for c in conditions} for k in a.k_shots} for name, _, _ in ARMS}
    feat_gen: dict[tuple[str, int], list[torch.Tensor]] = {}
    feat_real: list[torch.Tensor] = []
    grid_rows: dict[str, list[torch.Tensor]] = {n: [] for n, _, _ in ARMS}
    grid_real: list[torch.Tensor] = []
    k_grid = min(a.k_shots)

    total = len(conditions) * len(a.k_shots) * a.repeats
    done, t0 = 0, time.time()
    for ci, c in enumerate(conditions):
        others = [o for o in conditions if o != c]
        for k in a.k_shots:
            for r in range(a.repeats):
                b = loader.sample(k_shot=k, condition=c)
                real_c = real[c]
                real_sig = phototype_signature(real_c).mean(0)
                seed = a.seed + 1000 * ci + 100 * k + r      # shared across arms: paired
                with torch.no_grad():
                    z_wrong = enc(loader.sample(k_shot=1,
                                                condition=others[r % len(others)]).src_support)
                want_grid = (a.grid_out and k == k_grid and r == 0)
                if want_grid:
                    grid_real.append(real_c[:3].cpu())
                for name, strat, _ in ARMS:
                    ov = z_wrong if name == "wrong source" else None
                    oracle_data = real_c if strat == "oracle" else None
                    rgen = torch.Generator(device=dev).manual_seed(seed + 1)
                    st = adapt(strat, model, enc, tr, b, budget,
                               cfg.diffusion.loss_weighting, oracle_data=oracle_data,
                               z_s_override=ov, generator=rgen)
                    x = generate(model, st.z, a.n_samples, seed, a.ddim_steps, dev, size)
                    if want_grid:
                        grid_rows[name].append(x[:3].cpu())
                    cell = res[name][k][c]
                    sig = phototype_signature(x).mean(0)
                    cell["sig"].append(float((((sig - real_sig) / sig_sd) ** 2).sum().sqrt()))
                    cell["ita"].append(float(sig[0]))
                    fa, fb = x.flatten(1), real_c.flatten(1)
                    g = torch.Generator(device=dev).manual_seed(0)
                    cell["sw"].append(sliced_wasserstein(fa, fb, n_proj=256, generator=g))
                    cell["mmd"].append(energy_mmd(fa[:64], fb[:64]))
                    if ph_rep is not None:
                        cell["ph"].append(phenotype_target_share(ph_rep, x))
                        cell["cls"].append(condition_share(cd_rep, x, c))
                        with torch.no_grad():
                            feat_gen.setdefault((name, k), []).append(
                                cd_rep.model.features(x).cpu())
                done += 1
                if a.progress and done % a.progress == 0:
                    rate = (time.time() - t0) / done
                    print(f"  {done}/{total} cells, {rate:.1f}s each, "
                          f"~{rate * (total - done) / 60:.0f} min left",
                          file=sys.stderr, flush=True)
        if cd_rep is not None:
            with torch.no_grad():
                feat_real.append(cd_rep.model.features(real[c]).cpu())

    # ---- tables --------------------------------------------------------------------
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    def per_task(metric: str, title: str, note: str, higher: bool = False) -> None:
        emit(f"\n\n{title}")
        emit(f"  {note}")
        for c in conditions:
            emit(f"\n  -- {c} --")
            hdr = f"  {'arm':<18}" + "".join(f"{'K_T=' + str(k):>16}" for k in a.k_shots)
            emit(hdr)
            emit("  " + "-" * (len(hdr) - 3))
            for name, _, _ in ARMS:
                row = f"  {name:<18}"
                for k in a.k_shots:
                    mu, h = ci95(res[name][k][c][metric])
                    row += f"{mu:>10.3f} +-{h:<4.3f}" if not math.isnan(h) \
                        else f"{mu:>10.3f}      "
                emit(row)
        emit(f"\n  -- across the {len(conditions)} tasks, interval over tasks --")
        hdr = f"  {'arm':<18}" + "".join(f"{'K_T=' + str(k):>16}" for k in a.k_shots)
        emit(hdr)
        emit("  " + "-" * (len(hdr) - 3))
        for name, _, _ in ARMS:
            row = f"  {name:<18}"
            for k in a.k_shots:
                per = [float(np.mean(res[name][k][c][metric])) for c in conditions]
                mu, h = ci95(per)
                row += f"{mu:>10.3f} +-{h:<4.3f}"
            emit(row)

    per_task("sig", "distance from the real target population, in phototype-statistic space",
             "lower is better; the instrument is fixed (ITA, L*, b*, a*) and trains nothing")
    per_task("sw", "sliced Wasserstein distance to the real held-out target set, pixel space",
             "lower is better; two of the three tasks hold about 25 real images, so this is "
             "materially noisier than anything in stage A")
    per_task("ita", "the individual typology angle of the generated set",
             "not a distance: read it against the real target row printed above, and against "
             "the source row. higher = lighter skin")
    if ph_rep is not None:
        ceil = "  ".join(f"{k} {100*v:.1f}%" for k, v in ph_rep.accuracy.items())
        per_task("ph", "share of samples the population predictor calls Fitzpatrick IV-VI",
                 f"higher is better; the predictor's own ceiling: {ceil} (chance 50.0%)")
        cc = "  ".join(f"{k} {100*v:.1f}%" for k, v in cd_rep.accuracy.items())
        per_task("cls", "share of samples the condition predictor calls the right condition",
                 f"higher is better; ceiling {cc}, chance {100*cd_rep.chance:.1f}% -- a "
                 f"ceiling near chance means this row is uninformative, not that the arms agree")

    # ---- the plan's question, as paired differences over tasks ----------------------
    emit("\n\nthe claim: correct source plus learned relation against every way of not "
         "having it")
    emit("  paired within (task, K_T, resample); positive = transport is better")
    pairs = [("vs target only", "target only"), ("vs reuse z_S", "reuse z_S"),
             ("vs population only", "population only"), ("vs wrong source", "wrong source"),
             ("vs no adaptation", "no adaptation")]
    for metric, what in (("sig", "phototype-statistic space"),
                         ("sw", "sliced Wasserstein, pixel space")):
        emit(f"\n  {what}")
        hdr = f"  {'comparison':<20}" + "".join(f"{'K_T=' + str(k):>18}" for k in a.k_shots)
        emit(hdr)
        emit("  " + "-" * (len(hdr) - 3))
        for label, other in pairs:
            row = f"  {label:<20}"
            for k in a.k_shots:
                d = [x - y for c in conditions
                     for x, y in zip(res[other][k][c][metric], res["transport"][k][c][metric])]
                mu, h = ci95(d)
                row += f"{mu:>+11.4f} +-{h:<5.4f}{'*' if abs(mu) > h else ' '}"
            emit(row)
    emit("\n  * = the 95% interval over (task, resample) excludes zero. That interval is NOT "
         "a")
    emit("    per-task interval: with three tasks the between-task term dominates and is not")
    emit("    resolvable here. Read the per-task tables above and treat this as directional.")

    if feat_gen and feat_real:
        emit("\n\npooled KID against the real target sets, condition-instrument feature "
             "space (x1000)")
        emit("  lower is better; standard estimator, local feature space, so these compare "
             "arms here and nothing published")
        fr = torch.cat(feat_real)
        hdr = f"  {'arm':<18}" + "".join(f"{'K_T=' + str(k):>16}" for k in a.k_shots)
        emit(hdr)
        emit("  " + "-" * (len(hdr) - 3))
        kid_out: dict[str, dict[str, list[float]]] = {}
        for name, _, _ in ARMS:
            row, kid_out[name] = f"  {name:<18}", {}
            for k in a.k_shots:
                fg = torch.cat(feat_gen[(name, k)])
                gk = torch.Generator().manual_seed(a.seed)
                mu, h = kid(fg, fr, subset_size=min(100, fg.shape[0], fr.shape[0]),
                            n_subsets=100, generator=gk)
                kid_out[name][str(k)] = [mu, h, int(fg.shape[0]), int(fr.shape[0])]
                row += f"{1000*mu:>10.3f} +-{1000*h:<4.3f}"
            emit(row)
    else:
        kid_out = {}

    emit("\n\nwhat this can and cannot support")
    emit(f"  {len(conditions)} held-out tasks. Stage A had 60 episodes per K_T. A target-data")
    emit("  equivalence statement of the form 'the method at K_T=2 matches target-only at")
    emit("  K_T=10' needs a paired interval over held-out tasks and is not available at this")
    emit("  scale; a directional reading is. Only two things move the three: "
         "leave-one-condition-out")
    emit("  cross-validation (twelve training runs) or the authors' image release.")
    emit("  The population contrast is I-III against IV-VI, forced by what is reachable: "
         "under")
    emit("  I-II against V-VI the archive collapses to three conditions in total.")

    if a.json_out:
        out = a.json_out if os.path.isabs(a.json_out) else os.path.join(_ROOT, a.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        payload = {
            "checkpoint": os.path.basename(a.ckpt), "step": step, "split": a.split,
            "k_shots": a.k_shots, "repeats": a.repeats, "n_samples": a.n_samples,
            "ddim_steps": a.ddim_steps, "seed": a.seed, "image_size": size,
            "arm": cfg.model.score_model, "transport_kind": cfg.model.transport_kind,
            "split_checksum": split.checksum, "conditions": conditions,
            "real_signature": {c: phototype_signature(real[c]).mean(0).tolist()
                               for c in conditions},
            "source_signature": {c: phototype_signature(src_ref[c]).mean(0).tolist()
                                 for c in conditions},
            "n_real": {c: int(real[c].shape[0]) for c in conditions},
            "per_cell": {n: {str(k): res[n][k] for k in a.k_shots} for n, _, _ in ARMS},
            "kid_pooled": kid_out,
        }
        if ph_rep is not None:
            payload["instrument_ceilings"] = {
                "phenotype": ph_rep.accuracy, "condition": cd_rep.accuracy,
                "condition_n_way": cd_rep.n_classes}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        emit(f"\nper-cell values written to {a.json_out}")

    txt = os.path.splitext(a.json_out)[0] + ".txt" if a.json_out else None
    if txt:
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    if a.grid_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [n for n, _, _ in ARMS] + ["real target"]
        rows = [torch.cat(grid_rows[n], dim=0) for n, _, _ in ARMS] + [torch.cat(grid_real)]
        n_show = min(9, rows[0].shape[0])
        fig, axes = plt.subplots(len(rows), n_show,
                                 figsize=(n_show * 1.15, len(rows) * 1.28))
        for r, (lab, row) in enumerate(zip(labels, rows)):
            for cc in range(n_show):
                ax = axes[r, cc]
                ax.imshow(((row[cc].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if cc == 0:
                    ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
        fig.suptitle(f"one shared initial noise down each column, K_T = {k_grid}; "
                     "the bottom row is real held-out target data", fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.grid_out) or ".", exist_ok=True)
        fig.savefig(a.grid_out, dpi=130)
        print(f"\ngrid written to {a.grid_out}")


if __name__ == "__main__":
    main()
