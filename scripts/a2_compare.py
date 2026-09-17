"""A2: the additive reverse-dynamics basis against matched FiLM conditioning, at generation.

    python scripts/a2_compare.py --metric sw --out handoff/results/A2_basis_vs_film.txt

The two arms were trained from the same code path with the same seed, split, episode
construction, schedule, objective, optimiser and budget, and are evaluated with the same
episodes, the same source and target supports, the same initial noise and the same DDIM
configuration. So the episode lists line up and every difference below is paired.

The headline row is ``transport`` -- transport with no target-time refinement, which the
note asks for as the primary comparison because refinement has never shown a reliable
generation benefit on this project.

**What the floor is, and why it is not 'target only'.** A5 could lean on ``target only``
because the transport was the only thing that differed between its runs. Here the two arms
are two separately trained networks, so nothing that consults the coordinate is comparable
on architecture alone. What *is* comparable is ``no adaptation``: at z = 0 both arms reduce
exactly to the unconditioned U-Net -- the additive basis term vanishes and FiLM's
projections, which carry no bias, emit exactly zero. That row therefore differs between the
arms only by run-to-run variation of the same architecture, and its spread is the floor a
claim about the conditioning mechanism has to clear. One seed per arm cannot do better than
that; say so rather than implying otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DEFAULT_ARMS = [
    ("basis (Bz)", "handoff/results/A2_basis_per_episode.json"),
    ("FiLM", "handoff/results/A2_film_per_episode.json"),
]
ROWS = ["no adaptation", "reuse z_S", "transport", "transport + refine",
        "target only", "oracle"]
METRIC_NAMES = {
    "sw": "sliced Wasserstein to the real target set, pixel space (lower is better)",
    "sig": "distance in transformation-statistic space (lower is better)",
    "mmd": "energy MMD, pixel space (lower is better)",
    "cls": "share of samples in the episode's own class (HIGHER is better)",
    "sup": "the same at superclass level (HIGHER is better)",
}
HIGHER_IS_BETTER = {"cls", "sup"}


def ci95(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="sw", choices=tuple(METRIC_NAMES),
                    help="sw is the plan's headline; every metric present in both files "
                         "is also tabulated in the summary at the end")
    ap.add_argument("--basis", default=DEFAULT_ARMS[0][1])
    ap.add_argument("--film", default=DEFAULT_ARMS[1][1])
    ap.add_argument("--also", nargs="*", default=[], metavar="NAME=PATH",
                    help="further per-episode files to print absolute values for, e.g. "
                         "the cifar_ds3 checkpoint A1 and A3 were measured on. They are "
                         "not paired -- a different checkpoint means different episodes.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    def load(path: str):
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        if not os.path.exists(full):
            raise SystemExit(f"missing: {path}")
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)

    basis, film = load(a.basis), load(a.film)
    arms = [("basis (Bz)", basis), ("FiLM", film)]

    if basis["episodes"] != film["episodes"]:
        raise SystemExit("the two arms saw different episodes; they cannot be paired")
    for key in ("k_shots", "n_samples", "ddim_steps", "split", "m_source", "k"):
        if basis.get(key) != film.get(key):
            raise SystemExit(f"the arms disagree on '{key}': "
                             f"{basis.get(key)} against {film.get(key)} -- not matched")
    k_shots = basis["k_shots"]

    emit("A2 - additive reverse-dynamics basis against matched FiLM conditioning")
    emit("=" * 78)
    emit(f"{len(basis['episodes'])} held-out episodes, identical in both arms; "
         f"split {basis['split']}; k = {basis['k']}")
    emit(f"{basis['n_samples']} samples per cell, DDIM eta=0 with "
         f"{basis['ddim_steps']} steps, one shared initial noise per episode")
    emit(f"M_S = {basis['m_source']}, K_T in {k_shots}")
    emit("")
    for name, d in arms:
        emit(f"  {name:<12} {d['checkpoint']}  step {d['step']}  "
             f"{d['backbone']}"
             + (f" ({d['film_mode']})" if d.get("film_mode") else "")
             + f"  phi {d['n_params_phi']/1e6:.3f}M, "
             f"{d['n_params_conditioning']} parameters carry z")

    metric = a.metric
    emit("")
    emit("")
    emit(f"absolute readings -- {METRIC_NAMES[metric]}")
    head = f"\n  {'strategy':<20}" + "".join(
        f"{arm + ' K_T=' + str(k):^21}" for k in k_shots for arm, _ in arms)
    emit(head)
    emit("  " + "-" * (len(head) - 3))
    extra = [tuple(s.split("=", 1)) for s in a.also]
    for row in ROWS:
        s = f"  {row:<20}"
        for k in k_shots:
            for _, d in arms:
                vals = d["per_episode"].get(row, {}).get(str(k), {}).get(metric, [])
                if not vals:
                    s += f"{'-':>21}"
                    continue
                mu, h = ci95(vals)
                s += f"{mu:>13.4f} +-{h:<6.4f}"
        emit(s)
    for label, path in extra:
        d = load(path)
        # A1's and A5's files predate the arm/backbone fields, so read them softly
        shared_k = [k for k in d["k_shots"] if k in k_shots]
        emit(f"\n  not paired -- {label} ({d['checkpoint']}, "
             f"{len(d['episodes'])} episodes, {d.get('backbone', 'small_unet')})")
        emit(f"    {'':<18}" + "".join(f"{'K_T=' + str(k):^21}" for k in shared_k))
        for row in ROWS:
            vals_by_k = [d["per_episode"].get(row, {}).get(str(k), {}).get(metric, [])
                         for k in shared_k]
            if not any(vals_by_k):
                continue
            s = f"    {row:<18}"
            for vals in vals_by_k:
                mu, h = ci95(vals) if vals else (float("nan"), float("nan"))
                s += f"{mu:>13.4f} +-{h:<6.4f}"
            emit(s)

    emit("")
    emit("")
    emit("the question: does Bz transfer better than matched FiLM, episode by episode?")
    emit("  paired difference FiLM minus basis, signed so that positive = the basis is "
         "better")
    for m in [metric] + [x for x in METRIC_NAMES if x != metric]:
        if not basis["per_episode"]["transport"][str(k_shots[0])].get(m):
            continue
        sign = -1 if m in HIGHER_IS_BETTER else +1
        emit(f"\n  {m}: {METRIC_NAMES[m]}")
        emit(f"  {'strategy':<20}" + "".join(f"{'K_T=' + str(k):>20}" for k in k_shots))
        for row in ROWS:
            s = f"  {row:<20}"
            for k in k_shots:
                bv = basis["per_episode"][row][str(k)][m]
                fv = film["per_episode"][row][str(k)][m]
                d = [sign * (f - b) for f, b in zip(fv, bv)]
                mu, h = ci95(d)
                s += f"{mu:>+13.4f} +-{h:<5.4f}{'*' if abs(mu) > h else ' '}"
            emit(s)
    emit("\n  * = the 95% interval over episodes excludes zero")

    emit("")
    emit("")
    emit("the floor this has to clear")
    emit("  'no adaptation' is z = 0, where both arms reduce exactly to the unconditioned")
    emit("  U-Net: the basis term vanishes and FiLM's bias-free projections emit zero. Its")
    emit("  cross-arm difference is therefore run-to-run variation of one architecture, not")
    emit("  a conditioning effect, and a claim about the mechanism has to be larger than it.")
    for m in [metric]:
        for k in k_shots:
            bv = basis["per_episode"]["no adaptation"][str(k)][m]
            fv = film["per_episode"]["no adaptation"][str(k)][m]
            floor, fh = ci95([f - b for f, b in zip(fv, bv)])
            bt = basis["per_episode"]["transport"][str(k)][m]
            ft = film["per_episode"]["transport"][str(k)][m]
            eff, eh = ci95([f - b for f, b in zip(ft, bt)])
            ratio = abs(eff) / abs(floor) if abs(floor) > 1e-12 else float("inf")
            emit(f"    K_T={k:<3d} floor |{floor:+.4f}| +-{fh:.4f}    "
                 f"transport effect |{eff:+.4f}| +-{eh:.4f}    ratio {ratio:.1f}x")

    if basis.get("kid_pooled") and film.get("kid_pooled"):
        emit("")
        emit("")
        emit("pooled KID, instrument feature space, x1000 (lower is better)")
        emit("  not paired: the statistic is computed once on the pooled samples of a whole")
        emit("  cell, so it has a subset-to-subset interval and not an episode-to-episode one")
        emit(f"\n  {'strategy':<20}" + "".join(
            f"{'  ' + arm + ' K_T=' + str(k):>21}" for k in k_shots for arm, _ in arms))
        for row in ROWS:
            s = f"  {row:<20}"
            for k in k_shots:
                for _, d in arms:
                    cell = d["kid_pooled"].get(row, {}).get(str(k))
                    if cell is None:
                        s += f"{'-':>21}"
                        continue
                    s += f"{1000*cell[0]:>13.3f} +-{1000*cell[1]:<6.3f}"
            emit(s)
        n = film["kid_pooled"]["transport"][str(k_shots[0])]
        emit(f"\n  pooled from {n[2]} generated against {n[3]} real samples per cell")
        ceil = film.get("instrument_ceiling", {}).get("fine", {})
        if ceil:
            avg = 100 * sum(ceil.values()) / len(ceil)
            emit(f"  the instrument's own held-out accuracy, averaged over domains: "
                 f"{avg:.1f}% fine")

    emit("")
    emit("")
    emit("caveats that travel with this table")
    emit("  - one seed per arm. The four-seed sweep showed the transformation statistic's")
    emit("    seed spread routinely exceeding its own within-run half-width, so a marginal")
    emit("    row here is not settled until it is run again under other seeds.")
    emit("  - validation split. The test split is untouched and is spent once, at the end.")
    emit("  - the denoising-loss readings for both arms are in the training logs and are a")
    emit("    supporting diagnostic only; this table is the conclusion.")

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
