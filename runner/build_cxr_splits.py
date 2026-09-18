r"""Build the ChestX-ray14 task split. Refuses to overwrite an existing file.

    python -m runner.build_cxr_splits --out artifacts/chestxray_split_v2.json

Follows runner/build_cifar100_splits.py and runner/build_fitz_splits.py: the split file is
the contract, it is built once, and it stores stable identifiers -- image filenames -- rather
than paths or positions.

**Version 2, and why version 1 was withdrawn.** The first split (`chestxray_split.json`) was
audited against the plan's own text on 2026-09-18 and failed it in three places, each of which
this builder now enforces rather than leaves to chance:

1. *"pick age bins that give a source-rich / target-sparse relation"* -- v1 took 30-45 as the
   source, which is not the richest bin; 45-60 is (6 850 images against 4 077). The builder now
   refuses a source bin that is not the richest unless told otherwise, and prints the ranking.
2. *"Abundant source-population examples"* -- v1 used M_S = 16 with K_T up to 20, so at the top
   of the sweep the "scarce" target side outnumbered the "abundant" source (M_S/K_T = 0.8). It
   also allocated exactly M_S source images per evaluation task, so every episode showed the
   encoder the identical set. v2 uses M_S = 64, stage A's value, drawn per episode from a
   source **pool**.
3. *"enforce patient-level separation"* -- v1 separated patients across the population shift,
   the tasks, the relations and the splits, but not **between the K_T support and the held-out
   query of one (task, relation)**. All 30 pairs leaked: 1 168 query images were other films of
   a patient whose chest was in the support set, up to 34 % of one test query. That scores
   every arm that reads the support images partly against the images it conditioned on. v2
   allocates whole patients, and each K_T support image comes from a different patient, so
   K_T = 20 means twenty independent people.

The tasks are split train/validation/test -- which is what makes the relation rather than the
task the object of generalisation -- and there are several relations, which is the reason this
dataset replaced Fitzpatrick17k: with a single relation a constant coordinate solves the
meta-objective, and C4 measured exactly that.

Three contaminations are excluded before allocation, each measured by scripts/c0_chestxray.py:
multi-label rows; patients straddling an age bin (4.7 %); patients carrying more than one
finding (28.2 %), collapsed to their most frequent finding.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.chestxray import DEFAULT_ROOT, FINDINGS, load_annotations   # noqa: E402


def _stream_seed(seed: int, *parts: str | int) -> int:
    h = hashlib.sha256(("|".join(str(p) for p in (seed, *parts))).encode()).hexdigest()
    return int(h[:8], 16)


def allocate_source(images_by_patient: dict[str, list[str]], training: bool, m_source: int,
                    src_query: int, pool_cap: int, seed: int):
    """Whole patients only. Returns (support pool, query pool) or None if unfundable.

    For a training task the query takes whole patients until it holds src_query images, and
    the support pool is drawn from the remaining patients, so the two share no patient.
    """
    pats = sorted(images_by_patient)
    order = np.random.default_rng(seed).permutation(len(pats))
    pats = [pats[i] for i in order]
    query: list[str] = []
    i = 0
    if training:
        while i < len(pats) and len(query) < src_query:
            query += sorted(images_by_patient[pats[i]])
            i += 1
        if len(query) < src_query:
            return None
    support: list[str] = []
    while i < len(pats) and len(support) < pool_cap:
        support += sorted(images_by_patient[pats[i]])
        i += 1
    if len(support) < m_source:
        return None
    return support[:pool_cap], query


def allocate_target(images_by_patient: dict[str, list[str]], reserve: int, query_min: int,
                    seed: int):
    """One film from each of `reserve` distinct patients for the K_T support; the held-out
    query is every film of every *other* patient. Returns (reserve, query) or None."""
    pats = sorted(images_by_patient)
    if len(pats) < reserve + 1:
        return None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pats))
    pats = [pats[i] for i in order]
    res = []
    for p in pats[:reserve]:
        films = sorted(images_by_patient[p])
        res.append(films[int(rng.integers(len(films)))])
    query = [f for p in pats[reserve:] for f in sorted(images_by_patient[p])]
    if len(query) < query_min:
        return None
    return res, query


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="artifacts/chestxray_split_v2.json")
    ap.add_argument("--edges", type=int, nargs="+", default=[0, 30, 45, 60, 75, 96])
    ap.add_argument("--source-bin", type=int, default=2)
    ap.add_argument("--relation-bins", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--allow-non-richest-source", action="store_true",
                    help="the plan asks for a source-rich / target-sparse relation; without "
                         "this flag a source bin that is not the richest is refused")
    ap.add_argument("--m-source", type=int, default=64)
    ap.add_argument("--src-query", type=int, default=32)
    ap.add_argument("--src-pool-cap", type=int, default=300,
                    help="largest source support pool per task, CIFAR's src_support_size")
    ap.add_argument("--tgt-reserve", type=int, default=20)
    ap.add_argument("--tgt-query-min", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    if os.path.exists(out):
        raise SystemExit(
            f"{a.out} already exists. The split file is a contract: delete it deliberately "
            "or write elsewhere, but do not let a run silently rebuild it.")
    if a.m_source <= a.tgt_reserve:
        raise SystemExit(f"M_S={a.m_source} does not exceed max K_T={a.tgt_reserve}; the plan "
                         "asks for abundant source examples against scarce target ones")

    edges, src_bin = a.edges, a.source_bin
    n_bins = len(edges) - 1

    def binof(age: int) -> int:
        for i in range(n_bins):
            if edges[i] <= age < edges[i + 1]:
                return i
        return -1

    def bname(i: int) -> str:
        return f"{edges[i]}-{edges[i+1]}"

    ann = load_annotations(a.root)
    rows = list(ann.values())
    print(f"annotations: {len(rows)} rows, {len({r.patient for r in rows})} patients")

    # ---- the three exclusions ---------------------------------------------------------
    single = [r for r in rows if "|" not in r.finding and r.finding in FINDINGS]
    pat_bins: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        pat_bins[r.patient].add(binof(r.age))
    straddlers = {p for p, b in pat_bins.items() if len(b - {-1}) > 1}
    sel = [r for r in single if r.patient not in straddlers and binof(r.age) >= 0]
    per_pat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in sel:
        per_pat[r.patient][r.finding] += 1
    modal = {p: c.most_common(1)[0][0] for p, c in per_pat.items()}
    multi = sum(1 for c in per_pat.values() if len(c) > 1)
    sel = [r for r in sel if modal[r.patient] == r.finding]
    print(f"after filtering: {len(sel)} rows, {len({r.patient for r in sel})} patients "
          f"({len(straddlers)} bin-straddling dropped, {multi} multi-finding collapsed)")

    # ---- source-rich / target-sparse ---------------------------------------------------
    per_bin = collections.Counter(binof(r.age) for r in sel)
    ranking = sorted(range(n_bins), key=lambda i: -per_bin[i])
    print("images per age bin: " + ", ".join(f"{bname(i)} {per_bin[i]}" for i in ranking))
    if ranking[0] != src_bin and not a.allow_non_richest_source:
        raise SystemExit(
            f"source bin {bname(src_bin)} is not the richest ({bname(ranking[0])} is). The "
            "plan asks for a source-rich / target-sparse relation; pass "
            "--allow-non-richest-source to override, and say why in the write-up.")
    for t in a.relation_bins:
        if per_bin[t] >= per_bin[src_bin]:
            raise SystemExit(f"relation bin {bname(t)} is not sparser than the source")

    # ---- patient-grouped pools per (finding, bin) --------------------------------------
    pools: dict[tuple[str, int], dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in sel:
        pools[(r.finding, binof(r.age))][r.patient].append(r.filename)

    def try_task(f: str, training: bool):
        src = allocate_source(pools[(f, src_bin)], training, a.m_source, a.src_query,
                              a.src_pool_cap, _stream_seed(a.seed, f, "source"))
        if src is None:
            return None
        targets = {}
        for t in a.relation_bins:
            tg = allocate_target(pools[(f, t)], a.tgt_reserve, a.tgt_query_min,
                                 _stream_seed(a.seed, f, "target", t))
            if tg is None:
                return None
            targets[bname(t)] = {"bin": t, "tgt_support_reserve": tg[0], "tgt_query": tg[1]}
        return src, targets

    train_ok = [f for f in FINDINGS if try_task(f, True) is not None]
    eval_ok = [f for f in FINDINGS if try_task(f, False) is not None]
    print(f"usable tasks: {len(eval_ok)} eval-capable, {len(train_ok)} train-capable")
    if len(eval_ok) < a.n_train + a.n_val + 1:
        raise SystemExit(f"only {len(eval_ok)} usable tasks; asked for "
                         f"{a.n_train}/{a.n_val}/rest")

    def src_size(f: str) -> int:
        return sum(len(v) for v in pools[(f, src_bin)].values())

    train = sorted(sorted(train_ok, key=lambda f: -src_size(f))[:a.n_train])
    rest = [f for f in eval_ok if f not in train]
    rest = [rest[i] for i in np.random.default_rng(a.seed).permutation(len(rest))]
    val, test = sorted(rest[:a.n_val]), sorted(rest[a.n_val:])
    print(f"split: train {train}\n       val   {val}\n       test  {test}")

    tasks: dict[str, dict] = {}
    allocated = 0
    for which, names in (("train", train), ("val", val), ("test", test)):
        for f in names:
            (sup, qry), targets = try_task(f, which == "train")
            tasks[f] = {"split": which, "src_support": sup, "src_query": qry,
                        "targets": targets}
            allocated += len(sup) + len(qry) + sum(
                len(v["tgt_support_reserve"]) + len(v["tgt_query"]) for v in targets.values())

    payload = {
        "builder": "build_cxr_splits",
        "version": 2,
        "dataset": "chestxray14",
        "source": {"bin": src_bin, "name": bname(src_bin)},
        "relations": [{"bin": t, "name": bname(t)} for t in a.relation_bins],
        "config": {"seed": a.seed, "edges": edges, "m_source": a.m_source,
                   "src_query": a.src_query, "src_pool_cap": a.src_pool_cap,
                   "tgt_support_reserve": a.tgt_reserve, "tgt_query_min": a.tgt_query_min,
                   "patient_policy": "modal", "allocation": "whole patients; one K_T "
                   "support film per distinct patient; query from other patients only"},
        "split": {"train": train, "val": val, "test": test},
        "tasks": tasks,
        "excluded": {"straddling_patients": len(straddlers),
                     "multi_finding_patients": multi},
        "n_images": allocated,
    }
    payload["checksum"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\n{allocated} images allocated across {len(tasks)} tasks x "
          f"{len(a.relation_bins)} relations; M_S={a.m_source} from a pool of up to "
          f"{a.src_pool_cap}, max K_T={a.tgt_reserve}")
    print(f"written to {a.out}  checksum {payload['checksum'][:16]}...")


if __name__ == "__main__":
    main()
