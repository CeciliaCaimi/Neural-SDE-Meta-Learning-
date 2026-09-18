r"""C0 for NIH ChestX-ray14: the feasibility gate, from metadata alone.

    python scripts/c0_chestxray.py --out handoff/results/C0_chestxray.txt

The plan's fallback when Fitzpatrick17k fails: thoracic finding is the semantic task and age
group is the population split. This answers whether that is constructible **before** anything
is downloaded, because the 9 MB annotation table settles it and the images are 45 GB.

Three things it enforces, each of which changed the design when it was applied to
Fitzpatrick17k and would change it here:

1. **Single-positive-finding cases only**, as the plan asks. A row labelled
   "Effusion|Infiltration" belongs to no single task.
2. **Patient-level separation.** ChestX-ray14 has 30 805 patients and 112 120 images, a median
   of one image per patient but up to 184. Two images of one patient are not two independent
   samples, and if one lands in the source population and another in the target then part of
   the population shift is the same chest photographed twice -- the exact contamination that
   deduplication had to remove from Fitzpatrick17k.
3. **Patients that straddle age bins are dropped.** Follow-ups span up to 14 years, so 4.7 %
   of patients appear in two bins. Assigning them by modal age would keep them, but dropping
   them is unambiguous and costs little.

**The design question C4 answered the hard way.** Fitzpatrick offered one relation -- phototype
I-III to IV-VI -- and with one relation the same shift applies to every episode, so a constant
coordinate solves the meta-objective: the encoder's conditions ended up 1.4 % of a coordinate
norm apart and knowing the correct source task was worth nothing. Age is not binary. Several
target bins against one source bin give **several relations**, which is the structure CIFAR had
(three corruptions) and Fitzpatrick did not. This script therefore reports the task count at
each relation count, not just one table.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DEFAULT_ROOT = os.environ.get("CHESTXRAY_ROOT") or os.path.normpath(
    os.path.join(_ROOT, "..", "..", "..", "dataset", "chestxray14"))

# What a task has to fund, carried over from stage A and from Stage C's split builder.
M_S = 16              # source images the encoder sees; A4 showed saturation below sixteen
SRC_QUERY = 32        # source query batch, training tasks only
TGT_RESERVE = 20      # target support reserve, a nested prefix, so max K_T
TGT_QUERY_MIN = 24    # held-out target images a task needs to be measurable at all


def load(root: str):
    p = os.path.join(root, "Data_Entry_2017_v2020.csv")
    if not os.path.exists(p):
        raise SystemExit(
            f"{p} not found. Set CHESTXRAY_ROOT, or fetch the 9 MB annotation table:\n"
            "  https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset"
            "/resolve/main/data/Data_Entry_2017_v2020.csv")
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--edges", type=int, nargs="+", default=[0, 30, 45, 60, 75, 96],
                    help="age bin edges, half-open, chosen from the observed counts")
    ap.add_argument("--source-bin", type=int, default=1,
                    help="index of the source population bin; the rest are candidate targets")
    ap.add_argument("--patient-policy", choices=("modal", "drop"), default="modal",
                    help="what to do with a patient carrying more than one finding. They "
                         "leak across the TASK split, which is the split that makes the "
                         "relation rather than the task the object of generalisation. "
                         "'drop' removes them entirely and costs 48 %% of the rows; 'modal' "
                         "keeps each patient under their most frequent finding only, so no "
                         "patient appears under two tasks and nearly twice the data survives.")
    ap.add_argument("--include-no-finding", action="store_true",
                    help="treat 'No Finding' as a fifteenth task. It is 60 361 of the 91 324 "
                         "single-label rows, so it is a different kind of object from the "
                         "fourteen findings and is off by default.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    rows = load(a.root)
    edges = a.edges
    n_bins = len(edges) - 1

    def binof(age: int) -> int:
        for i in range(n_bins):
            if edges[i] <= age < edges[i + 1]:
                return i
        return -1

    emit("C0 -- NIH ChestX-ray14 as the Stage C population-shift dataset")
    emit("=" * 78)
    emit("From the annotation table alone. No images are needed to answer this.")
    emit(f"  {len(rows)} rows, {len({r['Patient ID'] for r in rows})} patients")

    # ---- 1. single-finding ---------------------------------------------------------
    single = [r for r in rows if "|" not in r["Finding Labels"]]
    emit(f"  single-positive-finding rows: {len(single)} "
         f"({100*len(single)/len(rows):.0f} %)")

    # ---- 2. patients that straddle a bin are dropped --------------------------------
    by_pat: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:                      # every row, not only single-finding: a patient
        by_pat[r["Patient ID"]].add(binof(int(r["Patient Age"])))   # straddles or it does not
    straddlers = {p for p, bs in by_pat.items() if len(bs - {-1}) > 1}
    emit(f"  patients straddling an age bin, dropped: {len(straddlers)} "
         f"({100*len(straddlers)/len(by_pat):.1f} %)")

    usable = [r for r in single
              if r["Patient ID"] not in straddlers and binof(int(r["Patient Age"])) >= 0]
    if not a.include_no_finding:
        usable = [r for r in usable if r["Finding Labels"] != "No Finding"]

    # A patient with two findings leaks across the TASK split: the model trains on that
    # chest under one finding and is evaluated on it under another.
    per_pat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in usable:
        per_pat[r["Patient ID"]][r["Finding Labels"]] += 1
    multi = {q for q, c in per_pat.items() if len(c) > 1}
    emit(f"  patients carrying more than one finding: {len(multi)} of {len(per_pat)} "
         f"({100*len(multi)/max(1,len(per_pat)):.1f} %)")
    before = len(usable)
    if a.patient_policy == "drop":
        usable = [r for r in usable if r["Patient ID"] not in multi]
    else:
        modal = {q: c.most_common(1)[0][0] for q, c in per_pat.items()}
        usable = [r for r in usable if modal[r["Patient ID"]] == r["Finding Labels"]]
    emit(f"  patient policy '{a.patient_policy}': {before} -> {len(usable)} rows")
    emit(f"  rows surviving every filter: {len(usable)}")
    emit(f"  patients surviving: {len({r['Patient ID'] for r in usable})}")

    # ---- 3. the finding-by-age-bin table --------------------------------------------
    findings = sorted({r["Finding Labels"] for r in usable})
    tab = {f: [0] * n_bins for f in findings}
    pats = {f: [set() for _ in range(n_bins)] for f in findings}
    for r in usable:
        b = binof(int(r["Patient Age"]))
        tab[r["Finding Labels"]][b] += 1
        pats[r["Finding Labels"]][b].add(r["Patient ID"])

    emit("")
    emit("")
    emit("finding by age bin -- images, and distinct patients in brackets")
    hdr = f"  {'finding':<22}" + "".join(
        f"{str(edges[i]) + '-' + str(edges[i+1]):>16}" for i in range(n_bins))
    emit(hdr)
    emit("  " + "-" * (len(hdr) - 3))
    for f in findings:
        emit(f"  {f:<22}" + "".join(
            f"{tab[f][i]:>9} ({len(pats[f][i]):>4})" for i in range(n_bins)))
    emit(f"  {'TOTAL':<22}" + "".join(
        f"{sum(tab[f][i] for f in findings):>9}       " for i in range(n_bins)))

    # ---- 4. how many tasks does each relation structure fund? -----------------------
    src = a.source_bin
    targets = [i for i in range(n_bins) if i != src]
    emit("")
    emit("")
    emit(f"source population: bin {src} = ages {edges[src]}-{edges[src+1]}")
    emit("Each remaining bin is one relation. The count that matters is not images but")
    emit("TASKS -- a task is a finding, and the conditions are split train/val/test so that")
    emit("the relation rather than the task is what generalises.")
    emit("")
    emit(f"  a task needs, per its own source side: M_S={M_S} (+{SRC_QUERY} query if training)")
    emit(f"  and per relation on the target side:   {TGT_RESERVE} reserve + "
         f"{TGT_QUERY_MIN} held-out query = {TGT_RESERVE + TGT_QUERY_MIN}")
    emit("")

    def qualifies(f: str, rels: list[int], training: bool) -> bool:
        need_src = M_S + (SRC_QUERY if training else 0)
        if tab[f][src] < need_src:
            return False
        return all(tab[f][t] >= TGT_RESERVE + TGT_QUERY_MIN for t in rels)

    emit(f"  {'relations used':<34}{'train-capable':>15}{'eval-capable':>14}"
         f"{'suggested split':>18}")
    emit("  " + "-" * 78)
    best = None
    for k in range(1, len(targets) + 1):
        rels = targets[:k]
        tr = [f for f in findings if qualifies(f, rels, True)]
        ev = [f for f in findings if qualifies(f, rels, False)]
        n_ev = len(ev)
        n_train = min(len(tr), max(0, n_ev - 6))
        n_val = (n_ev - n_train) // 2
        n_test = n_ev - n_train - n_val
        label = ",".join(f"{edges[t]}-{edges[t+1]}" for t in rels)
        emit(f"  {label:<34}{len(tr):>15}{n_ev:>14}"
             f"{f'{n_train}/{n_val}/{n_test}':>18}")
        # Prefer more relations while keeping at least three held-out test tasks. One
        # relation is what made C4 unanswerable, and more relations cost tasks, so the
        # choice is the largest relation count that still leaves a test side.
        # Prefer more relations, but only among options that are actually runnable: at
        # least four training tasks and three held-out test tasks. Without the first
        # condition this rule happily returned a 0/3/3 split, which trains on nothing.
        if (n_test >= 3 and n_train >= 4 and len(rels) >= 2
                and (best is None or len(rels) > len(best[0]))):
            best = (rels, tr, ev, n_train, n_val, n_test)

    emit("")
    emit("  'train-capable' also funds a source query batch, which only training tasks read.")
    emit("  The split is suggested, not fixed: the builder is what fixes it.")

    # ---- 5. the verdict --------------------------------------------------------------
    emit("")
    emit("")
    emit("verdict")
    if best is None:
        emit("  FAIL at this binning. No relation count of two or more leaves three held-out")
        emit("  test tasks. Try other --edges, or --patient-policy modal, before proceeding.")
    else:
        rels, tr, ev, n_train, n_val, n_test = best
        label = ", ".join(f"{edges[t]}-{edges[t+1]}" for t in rels)
        emit(f"  PASS. {len(rels)} relations ({label}) against source {edges[src]}-{edges[src+1]},")
        emit(f"  {len(ev)} usable tasks, suggested {n_train}/{n_val}/{n_test}.")
        emit(f"  Episodes per K_T on the test side: {n_test} tasks x {len(rels)} relations "
             f"= {n_test * len(rels)}.")
        emit("")
        emit("  Compare what this replaces. Fitzpatrick17k gave 3 test tasks and ONE relation,")
        emit("  and one relation is why C4 failed: the same shift applies to every episode, so")
        emit("  a constant coordinate solves the meta-objective and the correct source task is")
        emit("  worth nothing. Several relations is the structure CIFAR had, where the method")
        emit("  works. That, not the image count, is the reason to switch.")
        emit("")
        emit("  Still to check before trusting a headline, and none of it needs the images:")
        emit("    - between-task variance dominates, so count episodes as tasks x relations")
        emit("    - AP and PA views are different imaging geometries; a view imbalance across")
        emit("      age bins would be a confound riding along with the population shift")
        emit("    - sex is recorded and is a second candidate population axis")

    # ---- 6. the confound the plan does not mention ------------------------------------
    emit("")
    emit("")
    emit("view position by age bin, as a share of AP -- the confound to rule out")
    emit("  AP and PA are different geometries. If the source and target bins differ in view")
    emit("  mix, a model could score on the population shift by learning the geometry instead.")
    ap_share = []
    for i in range(n_bins):
        sel = [r for r in usable if binof(int(r["Patient Age"])) == i]
        share = sum(1 for r in sel if r["View Position"] == "AP") / max(1, len(sel))
        ap_share.append(share)
    emit(f"  {'bin':<22}" + "".join(
        f"{str(edges[i]) + '-' + str(edges[i+1]):>16}" for i in range(n_bins)))
    emit(f"  {'AP share':<22}" + "".join(f"{100*ap_share[i]:>15.1f}%" for i in range(n_bins)))
    spread = max(ap_share) - min(ap_share)
    emit(f"  spread across bins: {100*spread:.1f} percentage points")
    if spread > 0.15:
        emit("  LARGE. Restrict to one view, or report the view mix beside every verdict.")
    else:
        emit("  Small enough to report rather than design around.")

    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
