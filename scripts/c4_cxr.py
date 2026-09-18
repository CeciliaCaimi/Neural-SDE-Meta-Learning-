"""Stage C on ChestX-ray14: C3's sanity check, the fine-tuning budget selection, and C4.

    python scripts/c4_cxr.py CKPT --mode sanity                       # C3, validation tasks
    python scripts/c4_cxr.py CKPT --mode select-ft                    # budget for full_ft, val
    python scripts/c4_cxr.py CKPT --mode c4 --film-ckpt FILM --ft-budget FILE    # C4, test

One script, three modes, so that all three build episodes the same way, pair on the same
initial noise and read the same frozen instruments (checkpoints/cxr_instruments.pt).

**Pairing.** An episode cell is (task, relation, resample). Within a cell every arm and every
K_T shares the initial noise, the M_S source images and the held-out query, and the K_T support
is the nested prefix of one reserve -- so differences between arms at one K_T, and between one
arm at K_T = a and another at K_T = b, are both paired. The second is what the plan's headline
form needs.

**The arms** are the plan's C4 list plus the cross-cutting minimum baseline set (section 11):

    no adaptation      z = 0                                  (the run-to-run floor)
    target only        encode the K_T target films            (plan arm 1)
    reuse z_S          the source coordinate, no transport    (plan arm 2)
    relation only      Delta_gamma(0, c)                      (plan arm 3)
    wrong source       another task's source, same relation   (plan arm 4)
    transport          correct source + transport             (plan arm 5, the method)
    transport+refine   ... then refined on the K_T films      (section 11: with/without)
    FiLM transport     matched FiLM model, its own transport  (plan arm 6; section 11)
    full fine-tune     every weight, budget chosen on val     (section 11)
    oracle             refined on the abundant held-out set   (upper bound)

LoRA and adapters, which section 11 lists "if available", do not exist in this code base and
are not built; that is stated in the output rather than left for a reader to notice.

**The headline form is target-data equivalence** ("the method at K_T = 2 matches target-only
at K_T = 10"). The rule applied, fixed here before any result: transport at K_T = a *matches*
target-only at K_T = b when the paired difference transport@a - target@b is not established
worse, i.e. its 95 % interval over cells reaches zero or below. The largest such b is reported
for every a, and the whole a-by-b matrix is printed so a reader can apply a margin instead.

**The confound is reported beside every verdict.** Every table that reads predicted age prints
the AP share of the same generated sets, because the source and target age bins differ in view
mix by 14.7 points and the age instrument may partly read geometry.
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
from domains.chestxray import ZipIndex, load_chestxray                    # noqa: E402
from episodes.chestxray import CXRBatch, load_cxr_split                   # noqa: E402
from evaluation.cxr_instruments import ceiling_lines, load_instruments    # noqa: E402
from evaluation.metrics_analytic import energy_mmd, kid, sliced_wasserstein  # noqa: E402
from posthoc_controls import ci95, load_checkpoint                        # noqa: E402

METRICS = ("sw", "mmd", "age_gap")          # lower is better
READINGS = ("age", "age_src", "age_tgt", "ap", "cls")


@torch.no_grad()
def generate(model, z, n: int, seed: int, steps: int, dev) -> torch.Tensor:
    gen = torch.Generator(device=dev).manual_seed(seed)
    return ddim_sample(model.schedule, make_eps_fn(model, z), (n, 3, 32, 32), dev,
                       n_steps=steps, eta=0.0, generator=gen, clip_x0=1.0)


class Episodes:
    """Builds paired cells from the split directly, rather than through CXRLoader.sample,
    whose shared random stream would give each K_T a different source draw."""

    def __init__(self, split, raw, dev, m_source: int, seed: int):
        self.split, self.raw, self.dev, self.m_source, self.seed = split, raw, dev, m_source, seed
        self.images = torch.from_numpy(np.ascontiguousarray(raw.images)).to(dev)

    def fetch(self, names):
        gi = torch.from_numpy(self.raw.indices(list(names))).to(self.dev).long()
        return self.images.index_select(0, gi).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)

    def source(self, task: str, cell_seed: int):
        pool = self.split.tasks[task].src_support
        rng = np.random.default_rng(cell_seed)
        pick = rng.choice(len(pool), min(self.m_source, len(pool)), replace=False)
        return self.fetch([pool[int(i)] for i in sorted(pick)])

    def batch(self, task: str, rel: str, k: int, src) -> CXRBatch:
        tt = self.split.tasks[task].targets[rel]
        return CXRBatch(src_support=src, src_query=src,
                        tgt_support=self.fetch(tt["tgt_support_reserve"][:k]),
                        tgt_query=self.fetch(tt["tgt_query"]),
                        relation=torch.tensor(self.split.relation_names.index(rel),
                                              device=self.dev, dtype=torch.long),
                        provenance={"task": task, "relation": rel, "k": k})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--mode", choices=("sanity", "select-ft", "c4"), required=True)
    ap.add_argument("--split-path", default="artifacts/chestxray_split_v2.json")
    ap.add_argument("--instruments", default="checkpoints/cxr_instruments.pt")
    ap.add_argument("--film-ckpt", default=None)
    ap.add_argument("--ft-budget", default=None, help="JSON written by --mode select-ft")
    ap.add_argument("--k-shots", type=int, nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--progress", type=int, default=20)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = a.split_path if os.path.isabs(a.split_path) else os.path.join(_ROOT, a.split_path)
    split = load_cxr_split(split_path)
    with open(split_path, encoding="utf-8") as f:
        split_checksum = json.load(f)["checksum"]

    model, enc, tr, cfg, step, used_ema = load_checkpoint(a.ckpt, dev)
    if os.path.basename(cfg.episodes.cxr_path or "") != os.path.basename(split_path):
        raise SystemExit(f"{os.path.basename(a.ckpt)} was trained on "
                         f"{os.path.basename(cfg.episodes.cxr_path or '?')}, not on "
                         f"{os.path.basename(split_path)}; its episodes would not be held out")
    film = None
    if a.film_ckpt:
        film = load_checkpoint(a.film_ckpt, dev)
        if os.path.basename(film[3].episodes.cxr_path or "") != os.path.basename(split_path):
            raise SystemExit("the FiLM checkpoint was trained on a different split")
        if film[3].model.score_model != "film":
            raise SystemExit(f"{a.film_ckpt} is not a FiLM checkpoint "
                             f"(score_model={film[3].model.score_model})")
    inst = load_instruments(os.path.join(_ROOT, a.instruments), dev, split_checksum)

    # ---- mode defaults ------------------------------------------------------------
    if a.mode == "sanity":
        which, ks, reps = "val", a.k_shots or [20], a.repeats or 2
        arms = [("no adaptation", "zero"), ("reuse z_S", "source_reuse"),
                ("transport", "transport_no_refine"), ("oracle", "oracle")]
    elif a.mode == "select-ft":
        which, ks, reps = "val", a.k_shots or [1, 2, 5, 10, 20], a.repeats or 1
        arms = []
    else:
        which, ks, reps = "test", a.k_shots or [1, 2, 5, 10, 20], a.repeats or 4
        arms = [("no adaptation", "zero"), ("target only", "target_only"),
                ("reuse z_S", "source_reuse"), ("relation only", "relation_only"),
                ("wrong source", "transport_no_refine"), ("transport", "transport_no_refine"),
                ("transport+refine", "transport"), ("oracle", "oracle")]
        if film is not None:
            arms.append(("FiLM transport", "transport_no_refine"))
        ft_budget = None
        if a.ft_budget:
            with open(a.ft_budget, encoding="utf-8") as f:
                ft_budget = {int(k): v for k, v in json.load(f)["chosen"].items()}
            arms.append(("full fine-tune", "full_ft"))

    base_budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0,
                              noise_batch=cfg.adapt.noise_batch)
    ft_grid = [(s, lr) for s in (25, 100, 400) for lr in (1e-5, 1e-4)]
    if a.mode == "select-ft":
        arms = [(f"full_ft J={s} lr={lr:g}", "full_ft") for s, lr in ft_grid]

    tasks = split.names(which)
    rels = split.relation_names
    need = set()
    for t in tasks:
        ts = split.tasks[t]
        need |= set(ts.src_support)
        for r in rels:
            need |= set(ts.targets[r]["tgt_support_reserve"]) | set(ts.targets[r]["tgt_query"])
    raw = load_chestxray(names=sorted(need), index=ZipIndex(), verbose=False)
    ep = Episodes(split, raw, dev, cfg.episodes.enc_source_images, a.seed)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"Stage C on ChestX-ray14 -- mode {a.mode}")
    emit("=" * 78)
    emit(f"checkpoint {os.path.basename(a.ckpt)} step {step} ({'EMA' if used_ema else 'raw'}), "
         f"{cfg.model.backbone_kwargs.get('base_channels')} channels, k={cfg.model.k}, "
         f"transport {cfg.model.transport_kind}")
    if film is not None:
        emit(f"FiLM arm   {os.path.basename(a.film_ckpt)} step {film[4]}, trained identically")
    emit(f"split      {os.path.basename(split_path)} ({split_checksum[:12]}...), source "
         f"{split.source['name']}, relations {', '.join(rels)}")
    emit(f"episodes   {which} tasks {tasks} x {len(rels)} relations x {reps} resamples; "
         f"K_T in {ks}; M_S = {cfg.episodes.enc_source_images}; {a.n_samples} samples, "
         f"DDIM {a.ddim_steps} steps")
    for s in ceiling_lines(inst):
        emit(s)
    emit("LoRA / adapter baselines (section 11, 'if available'): not available in this code "
         "base and not built.")

    # ---- the sweep ------------------------------------------------------------------
    res: dict = {n: {k: {m: [] for m in METRICS + READINGS} for k in ks} for n, _ in arms}
    cells = []
    feats: dict = {}
    real_feats: list = []
    t0, done = time.time(), 0
    total = len(tasks) * len(rels) * reps
    for ti, task in enumerate(tasks):
        others = [o for o in tasks if o != task] or [o for o in split.names("train") if o != task]
        src_real_age = float(inst.predicted_age(ep.fetch(split.tasks[task].src_support)).mean())
        for ri, rel in enumerate(rels):
            for r in range(reps):
                cell_seed = a.seed + 1000 * ti + 100 * ri + r
                src = ep.source(task, cell_seed)
                wrong_src = ep.source(others[r % len(others)], cell_seed + 7)
                cells.append({"task": task, "relation": rel, "resample": r})
                for k in ks:
                    b = ep.batch(task, rel, k, src)
                    real = b.tgt_query
                    if k == ks[0] and r == 0:
                        real_feats.append(inst.features(real).cpu())
                    real_age = float(inst.predicted_age(real).mean())
                    fa_real = real.flatten(1)
                    for name, strat in arms:
                        rgen = torch.Generator(device=dev).manual_seed(cell_seed + 1)
                        m, e, t_ = model, enc, tr
                        kw = {}
                        bud = base_budget
                        if name == "FiLM transport":
                            m, e, t_ = film[0], film[1], film[2]
                        if name == "wrong source":
                            with torch.no_grad():
                                kw["z_s_override"] = enc(wrong_src)
                        if strat == "oracle":
                            kw["oracle_data"] = real
                        if strat == "full_ft":
                            if a.mode == "select-ft":
                                s_, lr_ = ft_grid[[n for n, _ in arms].index(name)]
                            else:
                                s_, lr_ = ft_budget[k]["steps"], ft_budget[k]["lr_weights"]
                            bud = base_budget.replace(steps=s_, lr_weights=lr_)
                        st = adapt(strat, m, e, t_, b, bud, cfg.diffusion.loss_weighting,
                                   generator=rgen, **kw)
                        gm = st.model if st.model is not None else m
                        x = generate(gm, st.z, a.n_samples, cell_seed, a.ddim_steps, dev)
                        c = res[name][k]
                        g = torch.Generator(device=dev).manual_seed(0)
                        c["sw"].append(sliced_wasserstein(x.flatten(1), fa_real, n_proj=256,
                                                          generator=g))
                        c["mmd"].append(energy_mmd(x.flatten(1)[:64], fa_real[:64]))
                        age_g = float(inst.predicted_age(x).mean())
                        c["age"].append(age_g)
                        c["age_src"].append(src_real_age)
                        c["age_tgt"].append(real_age)
                        c["age_gap"].append(abs(age_g - real_age))
                        c["ap"].append(inst.ap_share(x))
                        c["cls"].append(inst.finding_share(x, task))
                        feats.setdefault((name, k), []).append(inst.features(x).cpu())
                        if st.model is not None:
                            del st
                done += 1
                if a.progress and done % max(1, a.progress // max(1, len(ks))) == 0:
                    rate = (time.time() - t0) / done
                    print(f"  {done}/{total} cells, {rate:.0f}s each, "
                          f"~{rate * (total - done) / 60:.0f} min left",
                          file=sys.stderr, flush=True)

    names = [n for n, _ in arms]

    def cell_ci(name, k, m):
        return ci95(res[name][k][m])

    # ---- the tables ----------------------------------------------------------------------
    def table(metric: str, title: str, note: str) -> None:
        emit("")
        emit(title)
        emit(f"  {note}")
        hdr = f"  {'arm':<22}" + "".join(f"{'K_T=' + str(k):>17}" for k in ks)
        emit(hdr)
        emit("  " + "-" * (len(hdr) - 3))
        for n in names:
            emit(f"  {n:<22}" + "".join(
                f"{cell_ci(n, k, metric)[0]:>10.4f} +-{cell_ci(n, k, metric)[1]:<5.4f}"
                for k in ks))

    table("sw", "sliced Wasserstein to the real held-out target set, pixel space",
          "lower is better; the plan's main small-sample metric")
    table("age_gap", "|mean predicted age of the generated set - of the real target set|, years",
          "lower is better; read beside the AP share below, which is the confound")
    table("ap", "AP share of the generated set -- the confound, reported beside every age "
          "reading", "real source and target sets differ in AP share by up to 14.7 points")
    table("cls", "share of generated films the finding instrument assigns to the task's own "
          "finding", f"higher is better; instrument ceiling {100*inst.report['finding_acc']:.1f}%"
          f", chance {100*inst.report['finding_chance']:.1f}%")

    # the fraction of the real population shift a generated set reproduces, per relation
    emit("")
    emit("fraction of the real source-to-target age shift reproduced, pooled per relation")
    emit("  (mean gen - mean source) / (mean target - mean source), all on the age instrument;")
    emit("  1 = the whole shift, 0 = none. Pooled as a ratio of means, not a mean of ratios.")
    hdr = f"  {'arm':<22}{'relation':<9}" + "".join(f"{'K_T=' + str(k):>10}" for k in ks)
    emit(hdr)
    emit("  " + "-" * (len(hdr) - 3))
    for n in names:
        for rel in rels:
            row = f"  {n:<22}{rel:<9}"
            for k in ks:
                idx = [i for i, c in enumerate(cells) if c["relation"] == rel]
                g = [res[n][k]["age"][i] for i in idx]
                s = [res[n][k]["age_src"][i] for i in idx]
                t = [res[n][k]["age_tgt"][i] for i in idx]
                den = np.mean(t) - np.mean(s)
                row += f"{(np.mean(g) - np.mean(s)) / den:>10.2f}" if abs(den) > 0.5 \
                    else f"{'(tiny)':>10}"
            emit(row)

    if a.mode == "select-ft":
        chosen = {}
        emit("")
        emit("validation-selected budget for full fine-tuning, per K_T, on sliced Wasserstein")
        for k in ks:
            best = min(names, key=lambda n: cell_ci(n, k, "sw")[0])
            s_, lr_ = ft_grid[names.index(best)]
            chosen[k] = {"steps": s_, "lr_weights": lr_}
            emit(f"  K_T={k:<3} J={s_:<4} lr_weights={lr_:g}   "
                 f"sw {cell_ci(best, k, 'sw')[0]:.4f}")
        out = os.path.join(_ROOT, a.out_prefix or "handoff/results/C4_cxr_ft_budget")
        with open(out + ".json", "w", encoding="utf-8") as f:
            json.dump({"chosen": chosen, "grid": ft_grid, "selected_on": "val, sliced Wasserstein",
                       "checkpoint": os.path.basename(a.ckpt)}, f, indent=1)
        emit(f"\nwritten to {os.path.relpath(out, _ROOT)}.json")

    if a.mode in ("sanity", "c4"):
        emit("")
        emit("paired against transport, over (task, relation, resample) cells; "
             "positive = transport is better")
        for metric in ("sw", "age_gap"):
            emit(f"\n  {metric}")
            hdr = f"  {'comparison':<24}" + "".join(f"{'K_T=' + str(k):>19}" for k in ks)
            emit(hdr)
            emit("  " + "-" * (len(hdr) - 3))
            for n in names:
                if n == "transport":
                    continue
                row = f"  {'vs ' + n:<24}"
                for k in ks:
                    d = [x - y for x, y in zip(res[n][k][metric], res["transport"][k][metric])]
                    mu, h = ci95(d)
                    row += f"{mu:>+12.4f} +-{h:<5.4f}{'*' if abs(mu) > h else ' '}"
                emit(row)
        emit("\n  * = the 95% interval excludes zero. Cells are nested in tasks, and between-task")
        emit(f"    variance dominates; with {len(tasks)} {which} tasks read the per-task effect "
             "as directional.")

    if a.mode == "c4" and "target only" in names:
        emit("")
        emit("TARGET-DATA EQUIVALENCE -- the plan's headline form")
        emit("  transport at K_T=a matches target-only at K_T=b when transport@a - target@b is")
        emit("  not established worse: its 95% interval over cells reaches zero or below. Rule")
        emit("  fixed in the script's docstring before any result. Full matrix below it.")
        for metric in ("sw", "age_gap"):
            emit(f"\n  {metric}")
            for ka in ks:
                ok = []
                for kb in ks:
                    d = [x - y for x, y in zip(res["transport"][ka][metric],
                                               res["target only"][kb][metric])]
                    mu, h = ci95(d)
                    if mu - h <= 0:
                        ok.append(kb)
                verdict = (f"matches target-only up to K_T={max(ok)}" if ok
                           else "worse than target-only at every K_T")
                emit(f"    transport at K_T={ka:<3} {verdict}")
            emit("    matrix of paired means transport@a - target@b (rows a, columns b):")
            emit("      " + "".join(f"{'b=' + str(kb):>10}" for kb in ks))
            for ka in ks:
                emit(f"      a={ka:<3}" + "".join(
                    f"{np.mean([x - y for x, y in zip(res['transport'][ka][metric], res['target only'][kb][metric])]):>+10.4f}"
                    for kb in ks))

    if a.mode == "sanity":
        emit("")
        emit("C3 VERDICT -- does abundant target evidence move generation to the target group?")
        for rel in rels:
            idx = [i for i, c in enumerate(cells) if c["relation"] == rel]
            k = ks[-1]
            s = np.mean([res["oracle"][k]["age_src"][i] for i in idx])
            t = np.mean([res["oracle"][k]["age_tgt"][i] for i in idx])
            o = [res["oracle"][k]["age"][i] - res["reuse z_S"][k]["age"][i] for i in idx]
            mu, h = ci95(o)
            want = np.sign(t - s)
            moved = (np.sign(mu) == want) and abs(mu) > h
            emit(f"  relation {rel:<7} real source age {s:5.1f}, real target {t:5.1f}; oracle - "
                 f"reuse z_S = {mu:+.2f} +-{h:.2f} years -> "
                 f"{'MOVES towards the target' if moved else 'no established shift'}")
            ap_o = np.mean([res['oracle'][k]['ap'][i] for i in idx])
            ap_s = np.mean([res['reuse z_S'][k]['ap'][i] for i in idx])
            emit(f"  {'':<16}AP share oracle {100*ap_o:.1f}% vs reuse z_S {100*ap_s:.1f}% "
                 "(if this moved too, part of the age shift may be geometry)")

    # pooled KID
    if a.mode == "c4":
        emit("")
        emit("pooled KID, finding-instrument feature space (x1000); compares arms here only")
        fr = torch.cat(real_feats)
        emit(f"  {'arm':<22}" + "".join(f"{'K_T=' + str(k):>17}" for k in ks))
        for n in names:
            row = f"  {n:<22}"
            for k in ks:
                fg = torch.cat(feats[(n, k)])
                mu, h = kid(fg, fr, subset_size=min(100, fg.shape[0], fr.shape[0]),
                            n_subsets=100, generator=torch.Generator().manual_seed(a.seed))
                row += f"{1000*mu:>10.3f} +-{1000*h:<5.3f}"
            emit(row)

    prefix = a.out_prefix or {"sanity": "handoff/results/C3_cxr_sanity",
                              "select-ft": "handoff/results/C4_cxr_ft_select",
                              "c4": "handoff/results/C4_demographic_scarcity"}[a.mode]
    out = os.path.join(_ROOT, prefix)
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "checkpoint": os.path.basename(a.ckpt), "step": step,
                   "split_checksum": split_checksum, "k_shots": ks, "cells": cells,
                   "per_cell": {n: {str(k): res[n][k] for k in ks} for n in names}}, f)
    print(f"\nwritten to {prefix}.txt and .json")

    if a.mode == "c4":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        show = [n for n in ("target only", "reuse z_S", "transport", "FiLM transport",
                            "full fine-tune", "oracle") if n in names]
        for ax, metric, lab in ((axes[0], "sw", "sliced Wasserstein (lower is better)"),
                                (axes[1], "age_gap", "|age gap| in years (lower is better)")):
            for n in show:
                mu = [cell_ci(n, k, metric)[0] for k in ks]
                h = [cell_ci(n, k, metric)[1] for k in ks]
                ax.errorbar(ks, mu, yerr=h, marker="o", capsize=3, label=n,
                            lw=2.2 if n == "transport" else 1.2)
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlabel("K_T, target films available")
            ax.set_ylabel(lab)
        axes[0].legend(fontsize=7)
        fig.suptitle("C4 on ChestX-ray14: the scarcity curve, test tasks, 95% intervals over "
                     "cells", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(_ROOT, "handoff/results/C4_curve.png"), dpi=130)
        print("curve written to handoff/results/C4_curve.png")


if __name__ == "__main__":
    main()
