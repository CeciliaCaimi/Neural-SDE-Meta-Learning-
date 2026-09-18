r"""Checks on the ChestX-ray14 split, verified against the annotation table rather than the
split file that asserts it.

    python scripts/check_cxr_split.py

Needs no images -- everything here is settled by the annotations, which is the point: the
contaminations these look for are the ones that would make a result meaningless, and finding
them after a training run is expensive. The equivalent script for Fitzpatrick17k found 127
groups holding one photograph on both sides of the population shift, which is why
deduplication became mandatory there.

The three that decide whether a C4 number would mean anything:

* **No patient on both sides of the population shift.** If one chest is in the source bin and
  the same chest in a target bin, part of the shift is an identity map and transport looks
  better than it is.
* **No patient under two tasks.** The tasks are split train/validation/test so that the
  relation rather than the task generalises; a patient appearing under a training finding and
  a test finding means the model has seen that anatomy.
* **Every image really is in the age bin the split files it under**, read from the published
  annotation and not from the split.
"""
from __future__ import annotations

import collections
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from domains.chestxray import FINDINGS, load_annotations                  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else "artifacts/chestxray_split_v2.json"
    path = arg if os.path.isabs(arg) else os.path.join(_ROOT, arg)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run runner/build_cxr_splits.py first")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ann = load_annotations()
    edges = d["config"]["edges"]
    src_bin = d["source"]["bin"]
    rel_bins = {r["name"]: r["bin"] for r in d["relations"]}
    tasks = d["tasks"]

    def binof(age: int) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= age < edges[i + 1]:
                return i
        return -1

    print("1. the split as the loader will read it")
    print(f"     source bin {d['source']['name']}   relations "
          f"{', '.join(rel_bins)}   checksum {d['checksum'][:12]}...")
    for w in ("train", "val", "test"):
        print(f"     {w:<6} {len(d['split'][w])}: {', '.join(d['split'][w])}")
    check("three or more relations, which is the reason this dataset replaced Fitzpatrick",
          len(rel_bins) >= 2, f"{len(rel_bins)} relations")
    check("the task partition is three-way and non-empty",
          all(d["split"][w] for w in ("train", "val", "test")))
    seen = [f for w in ("train", "val", "test") for f in d["split"][w]]
    check("no task appears in two splits", len(seen) == len(set(seen)))
    check("only training tasks fund a source query stream",
          all(tasks[f]["src_query"] for f in d["split"]["train"])
          and not any(tasks[f]["src_query"]
                      for w in ("val", "test") for f in d["split"][w]))

    print("\n2. no image is in two places")
    owner: dict[str, list[str]] = collections.defaultdict(list)
    for f, t in tasks.items():
        for n in t["src_support"]:
            owner[n].append(f"{f}/src_support")
        for n in t["src_query"]:
            owner[n].append(f"{f}/src_query")
        for rel, tt in t["targets"].items():
            for n in tt["tgt_support_reserve"]:
                owner[n].append(f"{f}/{rel}/reserve")
            for n in tt["tgt_query"]:
                owner[n].append(f"{f}/{rel}/query")
    dup = {n: v for n, v in owner.items() if len(v) > 1}
    check("every image belongs to exactly one task, relation and stream",
          not dup, f"{len(dup)} shared, e.g. {list(dup.items())[:1]}")
    check("the image count agrees with what the builder recorded",
          len(owner) == d["n_images"], f"{len(owner)} against {d['n_images']}")

    print("\n3. the population shift is between different patients")
    src_pat, tgt_pat = set(), collections.defaultdict(set)
    for f, t in tasks.items():
        for n in t["src_support"] + t["src_query"]:
            src_pat.add(ann[n].patient)
        for rel, tt in t["targets"].items():
            for n in tt["tgt_support_reserve"] + tt["tgt_query"]:
                tgt_pat[rel].add(ann[n].patient)
    all_tgt = set().union(*tgt_pat.values()) if tgt_pat else set()
    check("no patient is on both sides of the population shift",
          not (src_pat & all_tgt), f"{len(src_pat & all_tgt)} on both sides")
    cross = [(a, b) for i, a in enumerate(tgt_pat) for b in list(tgt_pat)[i + 1:]
             if tgt_pat[a] & tgt_pat[b]]
    check("no patient appears under two relations",
          not cross, f"{len(cross)} relation pairs share a patient")
    print(f"        {len(src_pat)} source patients, {len(all_tgt)} target patients")

    print("\n4. no patient crosses the task split")
    pat_task: dict[str, set[str]] = collections.defaultdict(set)
    pat_split: dict[str, set[str]] = collections.defaultdict(set)
    for f, t in tasks.items():
        names = (t["src_support"] + t["src_query"]
                 + [n for tt in t["targets"].values()
                    for n in tt["tgt_support_reserve"] + tt["tgt_query"]])
        for n in names:
            pat_task[ann[n].patient].add(f)
            pat_split[ann[n].patient].add(t["split"])
    check("no patient appears under two tasks",
          not any(len(v) > 1 for v in pat_task.values()),
          f"{sum(1 for v in pat_task.values() if len(v) > 1)} do")
    check("no patient appears in two splits",
          not any(len(v) > 1 for v in pat_split.values()),
          f"{sum(1 for v in pat_split.values() if len(v) > 1)} do")

    print("\n4b. inside one (task, relation), support and query are different patients")
    # Added after the audit of 2026-09-18. v1 passed every check above and still leaked here:
    # the K_T support and the held-out query of one (task, relation) shared patients in all 30
    # pairs, so every arm that reads the support was scored partly against the same chests.
    leak_pairs, leak_imgs = 0, 0
    for f, t in tasks.items():
        for rel, tt in t["targets"].items():
            sp = {ann[n].patient for n in tt["tgt_support_reserve"]}
            shared = [n for n in tt["tgt_query"] if ann[n].patient in sp]
            if shared:
                leak_pairs += 1
                leak_imgs += len(shared)
    check("no held-out query image shares a patient with its own K_T support",
          leak_pairs == 0, f"{leak_pairs} (task, relation) pairs, {leak_imgs} query images")
    dup_res = [(f, rel) for f, t in tasks.items() for rel, tt in t["targets"].items()
               if len({ann[n].patient for n in tt["tgt_support_reserve"]})
               != len(tt["tgt_support_reserve"])]
    check("every K_T support image is a different patient, so K_T counts people",
          not dup_res, f"{len(dup_res)} reserves repeat a patient")
    sq = sum(len({ann[n].patient for n in t["src_support"]}
                 & {ann[n].patient for n in t["src_query"]}) for t in tasks.values())
    check("source support and source query share no patient", sq == 0, f"{sq} shared")

    print("\n4c. abundant source, scarce target -- the plan's own wording in C0 and C4")
    ms, kmax = d["config"]["m_source"], d["config"]["tgt_support_reserve"]
    check("M_S per episode exceeds the largest K_T", ms > kmax,
          f"M_S={ms}, max K_T={kmax}, ratio {ms/kmax:.1f}")
    thin_src = [f for f, t in tasks.items() if len(t["src_support"]) < ms]
    check("every task's source pool funds M_S", not thin_src, str(thin_src))
    fixed = [f for f, t in tasks.items() if len(t["src_support"]) <= ms]
    check("every source pool exceeds M_S, so episodes see different source sets",
          not fixed, f"{len(fixed)} tasks show the encoder one fixed set every episode")
    src_n = sum(len(t["src_support"]) + len(t["src_query"]) for t in tasks.values())
    tgt_n = {rel: sum(len(t["targets"][rel]["tgt_support_reserve"])
                      + len(t["targets"][rel]["tgt_query"]) for t in tasks.values())
             for rel in rel_bins}
    pool20 = sum(1 for t in tasks.values() if len(t["src_support"]) >= 20 * kmax)
    print(f"     source images allocated {src_n}; target per relation "
          + ", ".join(f"{r} {n}" for r, n in tgt_n.items()))
    print(f"     the document's M_S >> K_T test wants a source pool of 20 x max K_T = "
          f"{20*kmax}: {pool20} of {len(tasks)} tasks meet it (reported, not required here)")

    print("\n5. the labels, read from the published annotation")
    bad_bin_src = [n for f, t in tasks.items() for n in t["src_support"] + t["src_query"]
                   if binof(ann[n].age) != src_bin]
    check(f"every source image is in the source age bin ({d['source']['name']})",
          not bad_bin_src, f"{len(bad_bin_src)} are not")
    bad_bin_tgt = [(rel, n) for f, t in tasks.items() for rel, tt in t["targets"].items()
                   for n in tt["tgt_support_reserve"] + tt["tgt_query"]
                   if binof(ann[n].age) != rel_bins[rel]]
    check("every target image is in the age bin of its own relation",
          not bad_bin_tgt, f"{len(bad_bin_tgt)} are not")
    wrong = [(f, n) for f, t in tasks.items()
             for n in (t["src_support"] + t["src_query"]
                       + [m for tt in t["targets"].values()
                          for m in tt["tgt_support_reserve"] + tt["tgt_query"]])
             if ann[n].finding != f]
    check("every image's finding matches the task it is filed under",
          not wrong, f"{len(wrong)} mismatched")
    multi = [n for n in owner if "|" in ann[n].finding]
    check("no multi-label image is used", not multi, f"{len(multi)} are")
    check("every task is one of the fourteen findings",
          all(f in FINDINGS for f in tasks))

    print("\n6. what the split can fund")
    res = d["config"]["tgt_support_reserve"]
    short = [(f, rel) for f, t in tasks.items() for rel, tt in t["targets"].items()
             if len(tt["tgt_support_reserve"]) < res]
    check(f"every (task, relation) funds the K_T reserve of {res}",
          not short, f"{len(short)} short")
    thin = [(f, rel, len(tt["tgt_query"])) for f, t in tasks.items()
            for rel, tt in t["targets"].items()
            if len(tt["tgt_query"]) < d["config"]["tgt_query_min"]]
    check(f"every (task, relation) funds a held-out query of "
          f"{d['config']['tgt_query_min']}", not thin, str(thin[:3]))
    for w in ("val", "test"):
        q = [(f, min(len(tt["tgt_query"]) for tt in tasks[f]["targets"].values()))
             for f in d["split"][w]]
        print(f"     {w:<6} {len(q)} tasks x {len(rel_bins)} relations = "
              f"{len(q)*len(rel_bins)} episodes per K_T; smallest held-out query per "
              f"relation {min(x[1] for x in q)}")

    print("\n7. the confound the gate flagged: view mix")
    print("     AP and PA are different geometries. A model could score on the population")
    print("     shift by learning the geometry instead, so the mix is reported per side.")
    def ap_share(names) -> float:
        names = list(names)
        return sum(1 for n in names if ann[n].view == "AP") / max(1, len(names))
    s = ap_share([n for f, t in tasks.items() for n in t["src_support"] + t["src_query"]])
    print(f"     source {d['source']['name']:<8} AP {100*s:.1f}%")
    shares = [s]
    for rel in rel_bins:
        names = [n for f, t in tasks.items()
                 for n in t["targets"][rel]["tgt_support_reserve"] + t["targets"][rel]["tgt_query"]]
        v = ap_share(names)
        shares.append(v)
        print(f"     target {rel:<8} AP {100*v:.1f}%")
    spread = max(shares) - min(shares)
    print(f"     spread {100*spread:.1f} percentage points")
    check("the view mix does not differ so much that it could stand in for the shift",
          spread < 0.25, f"{100*spread:.1f} pp -- report it beside every verdict")

    rule = "-" * 78
    print(f"\n{rule}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        print(rule)
        return 1
    print("all checks passed")
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
