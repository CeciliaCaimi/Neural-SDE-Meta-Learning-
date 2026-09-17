"""How much of a reading is the seed?

    python scripts/seed_spread.py --out handoff/results/SEED_spread.txt

Every interval this project has reported is a within-run interval over episodes. It
contains the episode-to-episode component of the uncertainty and not the seed-to-seed
one, and the two are not the same size for every quantity: refinement's sign flipped
between two measurements whose half-widths said it should not have.

For each comparison this prints the paired difference at each seed, the within-seed
half-width, and the spread of the point estimates across seeds. **When the spread is
comparable to or larger than the half-widths, the within-run interval is not the honest
one and the direction should not be reported from a single seed.**
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

RUNS = [
    (4321, "handoff/results/A1_per_episode.json"),
    (1001, "handoff/results/SEED1001_per_episode.json"),
    (2002, "handoff/results/SEED2002_per_episode.json"),
    (3003, "handoff/results/SEED3003_per_episode.json"),
]

PAIRS = [
    ("transport vs reuse z_S", "reuse z_S", "transport"),
    ("transport vs target only", "target only", "transport"),
    ("transport vs mean z_S", "mean z_S", "transport"),
    ("transport vs shuffled z_S", "shuffled z_S", "transport"),
    ("transport vs relation only", "relation only", "transport"),
    ("refinement: off vs on", "transport", "transport + refine"),
    ("transport + refine vs oracle", "transport + refine", "oracle"),
]


def ci95(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sw", choices=("sw", "sig"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    data = {}
    for seed, path in RUNS:
        full = os.path.join(_ROOT, path)
        if not os.path.exists(full):
            emit(f"missing seed {seed}: {path}")
            continue
        with open(full, "r", encoding="utf-8") as f:
            data[seed] = json.load(f)

    seeds = sorted(data)
    k_shots = data[seeds[0]]["k_shots"]

    # The (class, transformation, K_T) labels are a fixed triple loop and are identical at
    # every seed by construction; what the seed changes is which images fill each episode
    # and which noise is drawn. So the check that the sweep swept anything has to be made
    # on the values, not on the labels.
    base = data[seeds[0]]["episodes"]
    ref_vals = data[seeds[0]]["per_episode"]["transport"][str(data[seeds[0]]["k_shots"][0])][a.metric]
    for s in seeds[1:]:
        vals = data[s]["per_episode"]["transport"][str(data[s]["k_shots"][0])][a.metric]
        if vals == ref_vals:
            raise SystemExit(f"seed {s} produced identical per-episode values to "
                             f"{seeds[0]}; the seed is not reaching the sampler")

    emit(f"metric: {a.metric}    seeds: {seeds}    episodes per run: {len(base)}")
    emit("the (class, transformation, K_T) labels are fixed by construction; the seed")
    emit("changes which images fill each episode and the noise, and the per-episode values")
    emit("differ between every pair of seeds, which is what makes this a sweep")

    for k in k_shots:
        emit(f"\n\nK_T = {k}")
        emit(f"  {'comparison':<28}" + "".join(f"{'seed ' + str(s):>12}" for s in seeds)
             + f"{'mean':>10}{'seed CI':>10}{'spread':>10}{'within-h':>11}{'verdict':>13}")
        emit("  " + "-" * (28 + 12 * len(seeds) + 47))
        for label, worse, better in PAIRS:
            mus, hs = [], []
            for s in seeds:
                d = [x - y for x, y in
                     zip(data[s]["per_episode"][worse][str(k)][a.metric],
                         data[s]["per_episode"][better][str(k)][a.metric])]
                mu, h = ci95(d)
                mus.append(mu)
                hs.append(h)
            spread = max(mus) - min(mus)
            mean_h = sum(hs) / len(hs)
            # The seed-level interval: each run is one independent draw of the whole
            # design, so the four point estimates are four observations and their standard
            # error is the honest uncertainty for an effect this small. This is the number
            # to report; the within-seed half-width is the one that misled.
            seed_mu = statistics.fmean(mus)
            seed_ci = (1.96 * statistics.stdev(mus) / math.sqrt(len(mus))
                       if len(mus) > 1 else float("nan"))
            same_sign = all(m > 0 for m in mus) or all(m < 0 for m in mus)
            # the verdict is decided on the seed-level interval, not the within-seed one
            if same_sign and abs(seed_mu) > seed_ci:
                verdict = "established"
            elif same_sign:
                verdict = "suggestive"
            else:
                verdict = "ZERO"
            row = f"  {label:<28}"
            for m in mus:
                row += f"{m:>+12.4f}"
            row += (f"{seed_mu:>+10.4f}{seed_ci:>10.4f}{spread:>10.4f}"
                    f"{mean_h:>11.4f}{verdict:>13}")
            emit(row)

    emit("\n\nhow to read this")
    emit("  seed CI     1.96 * SE over the four seed-level point estimates. This is the")
    emit("              number to quote: it carries the seed-to-seed component that the")
    emit("              within-seed half-width leaves out.")
    emit("  within-h    the mean of the four within-seed half-widths, shown only so the two")
    emit("              can be compared. Where 'spread' approaches it, a single run's")
    emit("              interval was never the honest one.")
    emit("  established same sign at all four seeds and the seed-level interval clear of zero")
    emit("  suggestive  same sign at all four seeds but the seed-level interval contains zero")
    emit("  ZERO        the seeds disagree on direction, so there is no effect to report")

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
