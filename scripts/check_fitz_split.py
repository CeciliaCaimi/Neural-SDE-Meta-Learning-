"""Checks on a Fitzpatrick split file. Every check raises rather than printing a warning.

    python scripts/check_fitz_split.py artifacts/fitzpatrick_split.json

The one that matters is the near-duplicate check. The builder removes duplicates as it
loads, so a bug there would produce a split that looks clean and leaks; this re-reads the
annotations from scratch and asserts against the file that was written, which is the only
version of the check that can fail for the right reason.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from runner.build_fitz_splits import components, near_duplicate_groups, side  # noqa: E402

STREAMS = ("src_support", "src_query", "tgt_support_reserve", "tgt_query")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--root", default=os.environ.get(
        "FITZPATRICK_ROOT", r"C:\T1D Meta Learning\dataset\fitzpatrick17k"))
    a = ap.parse_args()

    with open(a.split, encoding="utf-8") as f:
        sp = json.load(f)
    conds = sp["conditions"]
    cfg = sp["config"]
    print(f"{os.path.basename(a.split)}: {len(conds)} conditions, {sp['n_images']} images")
    print(f"M_S={cfg['m_source']} reserve={cfg['tgt_support_reserve']} "
          f"min query={cfg['tgt_query_min']} seed={cfg['seed']}\n")

    print("1. structure")
    parts = collections.Counter(c["split"] for c in conds.values())
    check("three splits present", set(parts) == {"train", "val", "test"}, str(dict(parts)))
    check("val and test hold the same number of conditions",
          parts["val"] == parts["test"], f"{parts['val']} vs {parts['test']}")
    for name, c in conds.items():
        if c["split"] == "train" and not c["src_query"]:
            check(f"training condition '{name}' has a source query batch", False)
        if c["split"] != "train" and c["src_query"]:
            check(f"evaluation condition '{name}' has no source query batch", False)
    check("training conditions carry a source query, evaluation ones do not",
          not any(f.startswith(("training condition", "evaluation condition"))
                  for f in FAILURES))

    all_ids = [h for c in conds.values() for s in STREAMS for h in c[s]]
    check("no image appears twice anywhere", len(all_ids) == len(set(all_ids)),
          f"{len(all_ids)} slots, {len(set(all_ids))} distinct")

    print("\n2. every image is on disk and on the side the split claims")
    got = {}
    with open(os.path.join(a.root, "manifest.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "cached"):
                got[r["md5hash"]] = r
    missing = [h for h in all_ids if h not in got]
    check("every allocated image was retrieved", not missing, f"{len(missing)} missing")

    truth = {}
    with open(os.path.join(a.root, "fitzpatrick17k.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            truth[r["md5hash"]] = (r["label"], side(r["fitzpatrick_scale"]))
    wrong_side = wrong_cond = 0
    for name, c in conds.items():
        for s in STREAMS:
            want = "source" if s.startswith("src") else "target"
            for h in c[s]:
                label, sd = truth.get(h, (None, None))
                if sd != want:
                    wrong_side += 1
                if label != name:
                    wrong_cond += 1
    check("source streams hold I-III and target streams hold IV-VI", wrong_side == 0,
          f"{wrong_side} misplaced")
    check("every image belongs to the condition it is filed under", wrong_cond == 0,
          f"{wrong_cond} misfiled")

    print("\n3. near duplicates")
    root_of = components(near_duplicate_groups(a.root))
    where = {}
    for name, c in conds.items():
        for s in STREAMS:
            for h in c[s]:
                where[h] = (name, c["split"], "source" if s.startswith("src") else "target")
    groups = collections.defaultdict(list)
    for h in where:
        if h in root_of:
            groups[root_of[h]].append(h)
    offending = {g: hs for g, hs in groups.items() if len(hs) > 1}
    check("no near-duplicate group has two images in the split", not offending,
          f"{len(offending)} groups survive" if offending else
          f"{len(groups)} groups represented, one image each")
    if offending:
        for g, hs in list(offending.items())[:5]:
            print("     ", [where[h] for h in hs])

    print("\n4. capacity, against the file's own thresholds")
    m_s, res, minq = cfg["m_source"], cfg["tgt_support_reserve"], cfg["tgt_query_min"]
    for name, c in conds.items():
        need_src = m_s + (cfg["src_query"] if c["split"] == "train" else 0)
        have_src = len(c["src_support"]) + len(c["src_query"])
        if have_src < need_src:
            check(f"'{name}' funds its source streams", False, f"{have_src} < {need_src}")
        if len(c["tgt_support_reserve"]) != res:
            check(f"'{name}' funds the support reserve", False)
        if len(c["tgt_query"]) < minq:
            check(f"'{name}' funds the held-out query", False, f"{len(c['tgt_query'])} < {minq}")
    check("every condition funds every stream it needs",
          not any(f.startswith("'") for f in FAILURES))

    print()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    print("all checks passed")


if __name__ == "__main__":
    main()
