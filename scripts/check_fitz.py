"""Does what actually landed on disk meet the plan's thresholds?

    python check_fitz.py

C0 built its tables from the annotation table, filtering by host and assuming every row
served by the reachable archive could be fetched. This script replaces that assumption with
the retrieval outcome recorded in the manifest, so the counts below are of files that exist.

Phototype grouping I-III against IV-VI, the grouping C0 selected from the observed counts.
Source population is the I-III side, target population the IV-VI side.
"""
from __future__ import annotations

import collections
import csv
import os

DEFAULT_ROOT = os.environ.get("FITZPATRICK_ROOT", r"C:\T1D Meta Learning\dataset\fitzpatrick17k")
DEST = DEFAULT_ROOT
CSV = os.path.join(DEFAULT_ROOT, "fitzpatrick17k.csv")

# The plan's floor, and what C4 needs on top of it.
FLOOR_NS, FLOOR_NT = 30, 15
KT_MAX, QUERY = 20, 25          # a K_T = 20 support set plus a held-out query set


def side(scale: str) -> str | None:
    try:
        v = int(scale)
    except ValueError:
        return None
    if 1 <= v <= 3:
        return "source"
    if 4 <= v <= 6:
        return "target"
    return None                  # -1, unknown phototype


def main() -> None:
    manifest = os.path.join(DEST, "manifest.csv")
    got = set()
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] in ("ok", "cached"):
                    got.add(r["md5hash"])
    print(f"manifest records {len(got)} retrieved files")

    with open(CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Three populations: what the full dataset promises, what the reachable archive
    # promises, and what is actually on disk.
    tables = {
        "full dataset": lambda r: True,
        "reachable archive": lambda r: "atlasdermatologico" in r["url"],
        "on disk": lambda r: r["md5hash"] in got,
    }

    counts: dict[str, dict[str, collections.Counter]] = {}
    for name, keep in tables.items():
        c = {"source": collections.Counter(), "target": collections.Counter()}
        for r in rows:
            s = side(r["fitzpatrick_scale"])
            if s and keep(r):
                c[s][r["label"]] += 1
        counts[name] = c

    print()
    print(f"{'population':<20}{'images':>10}{'conditions':>12}"
          f"{'N_S>=' + str(FLOOR_NS) + ' & N_T>=' + str(FLOOR_NT):>22}"
          f"{'can fund C4':>14}")
    print("-" * 78)
    summary = {}
    for name in tables:
        src, tgt = counts[name]["source"], counts[name]["target"]
        labels = set(src) | set(tgt)
        passes = [l for l in labels if src[l] >= FLOOR_NS and tgt[l] >= FLOOR_NT]
        fundable = [l for l in passes if tgt[l] >= KT_MAX + QUERY]
        n_img = sum(src.values()) + sum(tgt.values())
        summary[name] = (passes, fundable, src, tgt)
        print(f"{name:<20}{n_img:>10}{len(labels):>12}{len(passes):>22}{len(fundable):>14}")

    passes, fundable, src, tgt = summary["on disk"]
    print()
    print(f"conditions on disk passing the floor, sorted by target count "
          f"(C4 needs N_T >= {KT_MAX + QUERY})")
    print(f"  {'condition':<44}{'source':>9}{'target':>9}   fundable")
    for l in sorted(passes, key=lambda l: -tgt[l]):
        print(f"  {l[:44]:<44}{src[l]:>9}{tgt[l]:>9}   {'yes' if l in fundable else 'no'}")

    n = len(fundable)
    print()
    print(f"{n} conditions can fund a K_T = {KT_MAX} support set plus a {QUERY}-image "
          f"held-out query set.")
    if n:
        tr, va, te = max(1, int(n * 0.6)), max(1, int(n * 0.25)), 0
        te = n - tr - va
        print(f"Splitting those into train/validation/test task sets gives {tr}/{va}/{te}.")
        if te <= 1:
            print("A test set of one task cannot carry a paired interval over held-out tasks,")
            print("which is the form C4's headline result has to take.")


if __name__ == "__main__":
    main()
