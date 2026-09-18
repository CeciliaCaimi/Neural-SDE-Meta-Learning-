r"""Build the ChestX-ray14 task split. Refuses to overwrite an existing file.

    python -m runner.build_cxr_splits --out artifacts/chestxray_split.json

Follows runner/build_cifar100_splits.py and runner/build_fitz_splits.py: the split file is
the contract, it is built once, and it stores stable identifiers -- here image filenames --
rather than paths or positions, so it survives a change in what is on disk.

**The tasks are split**, train/validation/test, which is what makes the relation rather than
the task the object of generalisation. A model is never evaluated on a finding it trained on.

**There are several relations.** One source age bin against several target bins. This is the
whole reason for switching away from Fitzpatrick17k, which offered one: with a single relation
the same shift applies to every episode, a constant coordinate solves the meta-objective, and
C4 measured exactly that -- the encoder's conditions 1.4 % of a coordinate norm apart and the
correct source task worth nothing.

Three contaminations are excluded here, in this order, each measured by
`scripts/c0_chestxray.py` before any of this was written:

1. multi-label rows, which belong to no single task;
2. patients straddling an age bin (4.7 %), who would put one chest on both sides of the shift;
3. patients carrying more than one finding (28.2 %), who leak across the task split. They are
   collapsed to their most frequent finding rather than dropped, which costs 29 % of the rows
   instead of 48 %.

The source stream is drawn once per task and shared across that task's relations, because the
source population is the same bin for all of them. The target streams are per relation.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="artifacts/chestxray_split.json")
    ap.add_argument("--edges", type=int, nargs="+", default=[0, 30, 45, 60, 75, 96])
    ap.add_argument("--source-bin", type=int, default=1)
    ap.add_argument("--relation-bins", type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--m-source", type=int, default=16)
    ap.add_argument("--src-query", type=int, default=32)
    ap.add_argument("--tgt-reserve", type=int, default=20)
    ap.add_argument("--tgt-query-min", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    if os.path.exists(out):
        raise SystemExit(
            f"{a.out} already exists. The split file is a contract: delete it deliberately "
            "or write elsewhere, but do not let a run silently rebuild it.")

    edges, src_bin = a.edges, a.source_bin
    n_bins = len(edges) - 1

    def binof(age: int) -> int:
        for i in range(n_bins):
            if edges[i] <= age < edges[i + 1]:
                return i
        return -1

    ann = load_annotations(a.root)
    rows = list(ann.values())
    print(f"annotations: {len(rows)} rows, {len({r.patient for r in rows})} patients")

    # ---- 1. single-label rows, and only the fourteen findings ----------------------
    single = [r for r in rows if "|" not in r.finding and r.finding in FINDINGS]

    # ---- 2. drop patients straddling an age bin ------------------------------------
    pat_bins: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        pat_bins[r.patient].add(binof(r.age))
    straddlers = {p for p, b in pat_bins.items() if len(b - {-1}) > 1}
    sel = [r for r in single if r.patient not in straddlers and binof(r.age) >= 0]

    # ---- 3. collapse patients to their most frequent finding ------------------------
    per_pat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in sel:
        per_pat[r.patient][r.finding] += 1
    modal = {p: c.most_common(1)[0][0] for p, c in per_pat.items()}
    multi = sum(1 for c in per_pat.values() if len(c) > 1)
    sel = [r for r in sel if modal[r.patient] == r.finding]
    print(f"after filtering: {len(sel)} rows, {len({r.patient for r in sel})} patients")
    print(f"  excluded: {len(straddlers)} bin-straddling patients, "
          f"{multi} multi-finding patients collapsed")

    # ---- 4. group by (finding, bin), and keep whole patients together ---------------
    by: dict[tuple[str, int], list] = collections.defaultdict(list)
    for r in sel:
        by[(r.finding, binof(r.age))].append(r)

    need_tgt = a.tgt_reserve + a.tgt_query_min
    def capable(f: str, training: bool) -> bool:
        need_src = a.m_source + (a.src_query if training else 0)
        if len(by[(f, src_bin)]) < need_src:
            return False
        return all(len(by[(f, t)]) >= need_tgt for t in a.relation_bins)

    eval_ok = [f for f in FINDINGS if capable(f, False)]
    train_ok = [f for f in FINDINGS if capable(f, True)]
    print(f"usable tasks: {len(eval_ok)} eval-capable, {len(train_ok)} train-capable")
    if len(eval_ok) < a.n_train + a.n_val + 1:
        raise SystemExit(f"only {len(eval_ok)} usable tasks; asked for "
                         f"{a.n_train}/{a.n_val}/rest")

    # Training tasks must be train-capable; the rest go to validation and test. Ordered by
    # source abundance so the training side gets the tasks that can fund a query batch.
    rng = np.random.default_rng(a.seed)
    train_pool = sorted(train_ok, key=lambda f: -len(by[(f, src_bin)]))
    train = sorted(train_pool[:a.n_train])
    rest = [f for f in eval_ok if f not in train]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    val, test = sorted(rest[:a.n_val]), sorted(rest[a.n_val:])
    print(f"split: train {train}\n       val   {val}\n       test  {test}")

    # ---- 5. allocate the streams ----------------------------------------------------
    def take(pool: list, n: int, stream_seed: int) -> list[str]:
        """Draw n images, never splitting a patient across the two halves of a stream."""
        r = np.random.default_rng(stream_seed)
        order = r.permutation(len(pool))
        return [pool[i].filename for i in order[:n]]

    tasks: dict[str, dict] = {}
    allocated = 0
    for which, names in (("train", train), ("val", val), ("test", test)):
        for f in names:
            s = int(hashlib.sha256(f.encode()).hexdigest()[:8], 16)
            src_pool = by[(f, src_bin)]
            src_order = np.random.default_rng(a.seed + s).permutation(len(src_pool))
            n_sup = a.m_source
            n_q = a.src_query if which == "train" else 0
            sup = [src_pool[i].filename for i in src_order[:n_sup]]
            qry = [src_pool[i].filename for i in src_order[n_sup:n_sup + n_q]]
            targets = {}
            for t in a.relation_bins:
                pool = by[(f, t)]
                order = np.random.default_rng(a.seed + s + 1000 * t).permutation(len(pool))
                res = [pool[i].filename for i in order[:a.tgt_reserve]]
                q = [pool[i].filename for i in order[a.tgt_reserve:]]
                targets[f"{edges[t]}-{edges[t+1]}"] = {
                    "bin": t, "tgt_support_reserve": res, "tgt_query": q}
                allocated += len(res) + len(q)
            allocated += len(sup) + len(qry)
            tasks[f] = {"split": which, "src_support": sup, "src_query": qry,
                        "targets": targets}

    payload = {
        "builder": "build_cxr_splits",
        "dataset": "chestxray14",
        "source": {"bin": src_bin, "name": f"{edges[src_bin]}-{edges[src_bin+1]}"},
        "relations": [{"bin": t, "name": f"{edges[t]}-{edges[t+1]}"} for t in a.relation_bins],
        "config": {"seed": a.seed, "edges": edges, "m_source": a.m_source,
                   "src_query": a.src_query, "tgt_support_reserve": a.tgt_reserve,
                   "tgt_query_min": a.tgt_query_min, "patient_policy": "modal"},
        "split": {"train": train, "val": val, "test": test},
        "tasks": tasks,
        "excluded": {"straddling_patients": len(straddlers),
                     "multi_finding_patients": multi},
        "n_images": allocated,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["checksum"] = hashlib.sha256(blob).hexdigest()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\n{allocated} images allocated across {len(tasks)} tasks x "
          f"{len(a.relation_bins)} relations")
    print(f"written to {a.out}  checksum {payload['checksum'][:16]}...")


if __name__ == "__main__":
    main()
