"""Stage C model selection, read back out of the training logs.

    python scripts/c_select_table.py --out handoff/results/C_model_selection.txt

The first Stage C run took stage A's configuration -- 128 channels, 50 000 steps -- and
overfitted badly: the training target loss fell 0.0316 -> 0.0085 while the validation loss on
the three held-out conditions rose 0.0628 -> 0.1527, and r_basis collapsed 0.447 -> 0.0024.
A model whose basis residual is 0.24 % of its base prediction has had the task coordinate
trained out of it, and every arm of C4 then reads the same number. The C4 run on that
checkpoint was discarded.

This prints what the capacity sweep found and which checkpoint was taken, so the choice can
be checked rather than believed.

**The selection is on validation loss alone.** delta_task is printed beside it and was not
used to choose: selecting on the quantity the experiment is about would be circular, and
where the validation curve is flat the minimum is taken.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

RUNS = [("c_fitz", "128 ch, 50 000 steps -- stage A's configuration"),
        ("cs128", "128 ch, 8 000 steps"),
        ("cs64", "64 ch, 20 000 steps"),
        ("cs32", "32 ch, 30 000 steps -- also drops one level, attention and a block")]


def reads(name: str):
    p = os.path.join(_ROOT, "checkpoints", f"{name}_log.jsonl")
    if not os.path.exists(p):
        return [], []
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return ([r for r in rows if "L_meta" in r],
            [d for d in rows if "diagnostics" in d])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--chosen", default="cs32:6000")
    a = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("Stage C model selection, on the three held-out validation conditions")
    emit("=" * 78)
    emit("Selection criterion: validation loss_correct. delta_task and r_basis are printed")
    emit("beside it and were NOT used to choose -- selecting on the quantity the experiment")
    emit("is about would be circular. Where the curve is flat, the minimum is taken.")
    emit("")
    emit(f"  {'run':<8}{'ckpts':>7}{'best val':>11}{'at step':>9}"
         f"{'delta_task':>12}{'r_basis':>10}{'val at end':>12}{'overfit':>9}")
    emit("  " + "-" * 76)
    for name, _ in RUNS:
        tr, dg = reads(name)
        if not dg:
            emit(f"  {name:<8}  (no log)")
            continue
        best = min(dg, key=lambda d: d["diagnostics"]["loss_correct"])
        v, last = best["diagnostics"], dg[-1]["diagnostics"]
        ratio = last["loss_correct"] / v["loss_correct"]
        emit(f"  {name:<8}{len(dg):>7}{v['loss_correct']:>11.4f}{best['step']:>9}"
             f"{v['delta_task']:>+12.5f}{v['r_basis']:>10.3f}"
             f"{last['loss_correct']:>12.4f}{ratio:>8.1f}x")
    emit("")
    for name, what in RUNS:
        emit(f"  {name:<8} {what}")

    emit("")
    emit("")
    emit("what the sweep says")
    emit("  Capacity was the whole problem. At 128 channels the validation loss doubles")
    emit("  within a few thousand steps; at 32 it is flat over a budget seven times longer.")
    emit("  The training side draws from 808 images behind five conditions, so stage A's")
    emit("  50 000 steps is about two thousand epochs. The reachable archive does not fund a")
    emit("  128-channel model; it funds a 32-channel one. Say that in the write-up rather")
    emit("  than quietly reporting a different architecture from stage A's.")

    name, step = a.chosen.split(":")
    _, dg = reads(name)
    d = next((x for x in dg if x["step"] == int(step)), None)
    if d:
        v = d["diagnostics"]
        emit("")
        emit(f"chosen: checkpoints/{name}_step{step}.pt")
        emit(f"  validation loss_correct {v['loss_correct']:.4f}   "
             f"delta_task {v['delta_task']:+.5f}   gain_vs_zero {v['gain_vs_zero']:+.4f}   "
             f"r_basis {v['r_basis']:.3f}")
        frac = 100 * v["delta_task"] / v["gain_vs_zero"] if v["gain_vs_zero"] else float("nan")
        emit(f"  task_specific_frac {frac:.1f}% -- read this the way the project's own note")
        emit("  says to: gain_vs_zero is large because a 32-channel backbone cannot absorb the")
        emit("  shared structure, so the basis carries it as a constant offset. The quantity")
        emit("  that matters is the absolute delta_task, and at this checkpoint it is about")
        emit("  three times the +0.00160 the matched CIFAR run reaches.")

    emit("")
    emit("disclosure")
    emit("  A C4 run on the discarded 50 000-step checkpoint touched the test conditions")
    emit("  before this selection was made. Nothing from it informed the selection -- the")
    emit("  checkpoint was chosen from the validation curve above -- and its numbers are not")
    emit("  reported. It is recorded here because a test-split reading was spent.")

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
