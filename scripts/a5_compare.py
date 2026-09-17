"""A5: compare the four transport structures, episode by episode.

    python scripts/a5_compare.py

The four runs share a seed, a split and a protocol, so their episodes correspond one to
one and the comparison can be paired rather than made between two unrelated means.

The difficulty A5 has to survive is that these are four separately trained models, so a
difference between their transported coordinates mixes the map's structure with ordinary
run-to-run variation in the backbone and encoder they each trained. This script prints
the map-independent rows beside the map-dependent one for exactly that reason:
'target only' never consults the transport, so its spread across the four runs is a
floor on how large a difference has to be before the map can be credited with it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

RUNS = [
    ("residual_mlp", "handoff/results/A1_per_episode.json"),
    ("identity", "handoff/results/A5_identity_per_episode.json"),
    ("constant", "handoff/results/A5_constant_per_episode.json"),
    ("linear", "handoff/results/A5_linear_per_episode.json"),
]
PARAMS = {"residual_mlp": 6824, "identity": 0, "constant": 96, "linear": 352}


def ci95(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sw", choices=("sw", "sig", "mmd"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    data = {}
    for name, path in RUNS:
        full = os.path.join(_ROOT, path)
        if not os.path.exists(full):
            emit(f"missing: {path}")
            continue
        with open(full, "r", encoding="utf-8") as f:
            data[name] = json.load(f)

    ref = data["residual_mlp"]
    k_shots = ref["k_shots"]
    # the episodes must line up, or pairing across runs is meaningless
    for name, d in data.items():
        if d["episodes"] != ref["episodes"]:
            raise SystemExit(f"{name}: episode list differs from the reference; "
                             "these runs cannot be paired")
    emit(f"episodes identical across all {len(data)} runs: {len(ref['episodes'])}")
    emit(f"metric: {a.metric}   (sw = sliced Wasserstein, the plan's headline)")

    metric = a.metric
    emit("\nabsolute distance to the real target domain, lower is better")
    emit(f"  {'run':<14}{'params':>8}" + "".join(f"{'K_T=' + str(k):>18}" for k in k_shots))
    for row in ("transport", "target only", "oracle"):
        emit(f"\n  -- {row} --" + ("   (never consults the transport)"
                                   if row == "target only" else ""))
        for name, _ in RUNS:
            if name not in data:
                continue
            s = f"  {name:<14}{PARAMS[name]:>8}"
            for k in k_shots:
                mu, h = ci95(data[name]["per_episode"][row][str(k)][metric])
                s += f"{mu:>11.4f} +-{h:<5.4f}"
            emit(s)

    emit("\n\npaired against the residual MLP, episode by episode")
    emit("  positive = the residual MLP is better on that episode")
    for row in ("transport", "target only"):
        emit(f"\n  -- {row} --")
        emit(f"  {'run':<14}" + "".join(f"{'K_T=' + str(k):>19}" for k in k_shots))
        for name, _ in RUNS[1:]:
            if name not in data:
                continue
            s = f"  {name:<14}"
            for k in k_shots:
                d = [x - y for x, y in zip(data[name]["per_episode"][row][str(k)][metric],
                                           ref["per_episode"][row][str(k)][metric])]
                mu, h = ci95(d)
                s += f"{mu:>+12.4f} +-{h:<5.4f}{'*' if abs(mu) > h else ' '}"
            emit(s)

    emit("\n  * = the 95% interval excludes zero")

    emit("\n\nthe floor a map-structure claim has to clear")
    emit("  'target only' shares the episodes and never touches the transport, so the")
    emit("  spread of its readings across these runs is ordinary run-to-run variation.")
    for k in k_shots:
        tgt = [ci95(data[n]["per_episode"]["target only"][str(k)][metric])[0] for n, _ in RUNS if n in data]
        tr = [ci95(data[n]["per_episode"]["transport"][str(k)][metric])[0]
              for n, _ in RUNS if n in data and n != "identity"]
        emit(f"    K_T={k:<3d} target-only spread across runs {max(tgt) - min(tgt):.4f}   "
             f"transport spread, identity excluded {max(tr) - min(tr):.4f}")

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
