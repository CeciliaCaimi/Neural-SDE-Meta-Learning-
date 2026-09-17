"""The complete A1 table, assembled from the per-episode dumps of all four seeds.

    python scripts/a1_table.py --json handoff/results/A1_full_table.json

Emits every number the source-dependence panel produced: absolute distances for all nine
conditions, paired differences for every comparison, both metrics, all three values of
K_T, at each seed and pooled across seeds. Nothing is transcribed from the printed tables;
everything is recomputed from the episode-level values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SEEDS = [
    (4321, "handoff/results/A1_per_episode.json"),
    (1001, "handoff/results/SEED1001_per_episode.json"),
    (2002, "handoff/results/SEED2002_per_episode.json"),
    (3003, "handoff/results/SEED3003_per_episode.json"),
]

CONDITIONS = [
    ("no adaptation", "z = 0, nothing fitted"),
    ("reuse z_S", "the source coordinate, untransported"),
    ("transport", "the method: transport, no target image"),
    ("transport + refine", "transport, then the K_T target images"),
    ("target only", "encode the K_T images, no source"),
    ("oracle", "refined on abundant target data"),
    ("mean z_S", "mean source coordinate of this transformation"),
    ("shuffled z_S", "another class's source coordinate, same transformation"),
    ("relation only", "transport from the relation alone, z_S zeroed"),
]

PAIRS = [
    ("transport vs reuse z_S", "reuse z_S", "transport", "is transporting better than reusing"),
    ("transport vs target only", "target only", "transport", "is the source worth anything"),
    ("transport vs mean z_S", "mean z_S", "transport", "is THIS task's coordinate needed"),
    ("transport vs shuffled z_S", "shuffled z_S", "transport", "is the right one needed"),
    ("transport vs relation only", "relation only", "transport", "is any source coordinate needed"),
    ("refinement: off vs on", "transport", "transport + refine", "does refinement earn its place"),
    ("transport + refine vs oracle", "transport + refine", "oracle", "how far from the ceiling"),
]


def ci95(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="handoff/results/A1_full_table.json")
    a = ap.parse_args()

    data = {}
    for seed, path in SEEDS:
        full = os.path.join(_ROOT, path)
        if os.path.exists(full):
            with open(full, "r", encoding="utf-8") as f:
                data[seed] = json.load(f)
    seeds = sorted(data)
    ref = data[seeds[0]]
    k_shots = ref["k_shots"]
    print(f"seeds {seeds}, {len(ref['episodes'])} episodes each, K_T {k_shots}")

    out = {
        "checkpoint": ref["checkpoint"], "step": ref["step"], "split": ref["split"],
        "episodes": len(ref["episodes"]), "k_shots": k_shots, "seeds": seeds,
        "n_samples": ref["n_samples"], "ddim_steps": ref["ddim_steps"],
        "m_source": ref["m_source"],
        "conditions": [{"name": n, "what": w} for n, w in CONDITIONS],
        "absolute": {}, "paired": {},
    }

    for metric in ("sw", "sig"):
        out["absolute"][metric] = {}
        for name, _ in CONDITIONS:
            row = {}
            for k in k_shots:
                per_seed = [ci95(data[s]["per_episode"][name][str(k)][metric])[0] for s in seeds]
                mu, h = ci95(data[seeds[0]]["per_episode"][name][str(k)][metric])
                row[str(k)] = {
                    "seed4321": mu, "seed4321_ci": h,
                    "pooled": statistics.fmean(per_seed),
                    "pooled_ci": (1.96 * statistics.stdev(per_seed) / math.sqrt(len(per_seed))
                                  if len(per_seed) > 1 else None),
                    "per_seed": per_seed,
                }
            out["absolute"][metric][name] = row

        out["paired"][metric] = {}
        for label, worse, better, question in PAIRS:
            row = {"question": question}
            for k in k_shots:
                mus, hs = [], []
                for s in seeds:
                    d = [x - y for x, y in zip(data[s]["per_episode"][worse][str(k)][metric],
                                               data[s]["per_episode"][better][str(k)][metric])]
                    mu, h = ci95(d)
                    mus.append(mu)
                    hs.append(h)
                same_sign = all(m > 0 for m in mus) or all(m < 0 for m in mus)
                pooled = statistics.fmean(mus)
                pooled_ci = (1.96 * statistics.stdev(mus) / math.sqrt(len(mus))
                             if len(mus) > 1 else None)
                row[str(k)] = {
                    "seed4321": mus[seeds.index(4321)],
                    "seed4321_ci": hs[seeds.index(4321)],
                    "per_seed": mus, "pooled": pooled, "pooled_ci": pooled_ci,
                    "verdict": ("established" if same_sign and pooled_ci is not None
                                and abs(pooled) > pooled_ci
                                else "suggestive" if same_sign else "zero"),
                }
            out["paired"][metric][label] = row

    path = a.json if os.path.isabs(a.json) else os.path.join(_ROOT, a.json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("written to", a.json)

    # a readable echo, so the artefact can be checked against this
    for metric in ("sw", "sig"):
        print(f"\n=== paired, {metric} ===")
        for label, *_ in PAIRS:
            r = out["paired"][metric][label]
            s = f"  {label:<30}"
            for k in k_shots:
                c = r[str(k)]
                s += f"{c['pooled']:>+9.4f}+-{c['pooled_ci']:<7.4f}{c['verdict'][:4]:>5}"
            print(s)


if __name__ == "__main__":
    main()
