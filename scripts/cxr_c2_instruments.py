"""C2 for ChestX-ray14: build the frozen instruments once, report their ceilings, save them.

    python scripts/cxr_c2_instruments.py

Writes checkpoints/cxr_instruments.pt (every later evaluation loads it) and
handoff/results/C2_cxr_instruments.txt. See evaluation/cxr_instruments.py for why the
instruments are trained once and what is excluded from their training data.

Two readings here decide how far any Stage C verdict can be trusted:

* **Is predicted age monotone across the age bins on real held-out films?** This is the
  analogue of checking that ITA was monotone across the six skin types. An age instrument
  that cannot order real films by age cannot say whether a generated set moved towards a
  target age group.
* **Does predicted age differ between AP and PA films of the same true age bin?** If it does,
  the instrument partly reads imaging geometry, and a shift in the view mix of a generated set
  would register as an age shift. That is the confound the audit found.
"""
from __future__ import annotations

import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evaluation.cxr_instruments import (                                  # noqa: E402
    build_instruments, ceiling_lines, save_instruments,
)


def main() -> None:
    split_path = os.path.join(_ROOT, "artifacts", "chestxray_split_v2.json")
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inst = build_instruments(split, device=dev, seed=4321)
    r = inst.report
    lines = ["C2 -- frozen evaluation instruments for ChestX-ray14", "=" * 78, ""]
    lines += ceiling_lines(inst)

    lines += ["", "predicted age by true age bin, on held-out patients -- must rise with age",
              f"  {'true bin':<10}{'all':>16}{'PA':>16}{'AP':>16}{'AP - PA':>12}"]
    bins = sorted({k.split('|')[0] for k in r["age_by_bin_and_view"]},
                  key=lambda b: int(b.split('-')[0]))
    prev, monotone, gaps = None, True, []
    for b in bins:
        cell = {v: r["age_by_bin_and_view"].get(f"{b}|{v}") for v in ("all", "PA", "AP")}
        txt = "".join(f"{c[0]:>9.1f} (n{c[1]:>4})" if c else f"{'-':>16}"
                      for c in (cell["all"], cell["PA"], cell["AP"]))
        gap = (cell["AP"][0] - cell["PA"][0]) if cell["AP"] and cell["PA"] else None
        if gap is not None:
            gaps.append(gap)
        lines.append(f"  {b:<10}{txt}{(f'{gap:+12.1f}') if gap is not None else '':>12}")
        if cell["all"] and prev is not None and cell["all"][0] <= prev:
            monotone = False
        if cell["all"]:
            prev = cell["all"][0]
    lines.append("")
    lines.append(f"  monotone across bins: {'YES' if monotone else 'NO'}")
    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        lines.append(f"  mean AP - PA gap at equal true age: {mean_gap:+.1f} years -- the "
                     "confound in the instrument itself. A generated set whose view mix moves")
        lines.append("  by d reads as an age shift of roughly d x this gap; every verdict prints "
                     "the AP share beside the age reading so the two can be separated.")

    lines += ["", "finding classifier, per class on held-out patients (chance "
              f"{100*r['finding_chance']:.1f}%)"]
    for f_, (acc, n) in sorted(r["finding_per_class"].items(), key=lambda kv: -kv[1][0]):
        mark = "   <- a test task" if f_ in split["split"]["test"] else ""
        lines.append(f"  {f_:<22}{100*acc:6.1f}%  (n {n}){mark}")

    ck = os.path.join(_ROOT, "checkpoints", "cxr_instruments.pt")
    save_instruments(ck, inst)
    lines += ["", f"saved to checkpoints/cxr_instruments.pt; every later evaluation loads this "
              "file and refuses it if the split checksum differs"]
    out = os.path.join(_ROOT, "handoff", "results", "C2_cxr_instruments.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
