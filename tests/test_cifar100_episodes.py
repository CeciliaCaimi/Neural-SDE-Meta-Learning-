"""Loading verification for the split.

    python -m tests.test_cifar100_episodes

Walk **every** episode and every K_T, checking image by image:
  1. does the split file round-trip through disk (save -> load yields the same object)?
  2. can every image actually be fetched, with the right shape and dtype?
  3. does each loaded image carry the fine label of the class it was assigned to?
  4. does each loaded image carry the coarse label of its episode (= the relation)?
  5. are the four index streams pairwise disjoint?
  6. is the target support nested over K_T?
  7. do semantic classes avoid leaking across splits?
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.cifar100 import load_cifar100                                # noqa: E402
from episodes.guards import check_episode_labels, check_split             # noqa: E402
from episodes.sampler import iter_episodes, make_episode                  # noqa: E402
from episodes.splits import ROLE_TARGET, load_split                       # noqa: E402

SPLIT_PATH = os.path.join(_ROOT, "artifacts", "cifar100_split.json")
RULE = "─" * 78
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ——  {detail}" if detail else ""))


def main() -> int:
    print(RULE)
    print("split loading verification")
    print(RULE)

    raw = load_cifar100()
    split = load_split(SPLIT_PATH)
    print(f"split file : {SPLIT_PATH}")
    print(f"raw data   : {raw.images.shape[0]} images\n")

    # ---- 1. Round trip ----
    print("[1] split file round trip")
    check_split(split)
    n_pools = len(split.pools)
    n_idx = sum(
        p.src_support.size + p.src_query.size + p.tgt_support_reserve.size + p.tgt_query.size
        for p in split.pools.values()
    )
    record("check_split still passes after reloading from disk", True, f"{n_pools} class pools, {n_idx} image ids")
    record("every image is assigned exactly once", n_idx == 60000, f"{n_idx} == 60000")

    # ---- 2-5. Full episode walk ----
    print("\n[2] full episode walk (labels checked image by image)")
    t0 = time.time()
    n_ep = n_img = 0
    shape_ok = dtype_ok = True
    for which in ("train", "val", "test"):
        for k in split.config.k_shots:
            for ep in iter_episodes(split, which, k_shot=k):
                check_episode_labels(ep, raw)          # assertions 3 and 4; raises LeakageError
                for name in ep.all_index_sets():
                    px = ep.materialise(raw, name)
                    shape_ok &= px.shape[1:] == (32, 32, 3)
                    dtype_ok &= px.dtype == np.uint8
                    n_img += px.shape[0]
                n_ep += 1
    dt = time.time() - t0
    record("all episodes pass the label consistency check", True, f"{n_ep} episodes, {n_img:,} images, {dt:.1f}s")
    record("every image has shape (32,32,3)", shape_ok)
    record("every image has dtype uint8", dtype_ok)

    # ---- 6. K_T nesting ----
    print("\n[3] K_T nesting of the target support")
    ks = list(split.config.k_shots)
    nested = True
    example = ""
    for fine_id in split.fine_ids("train", ROLE_TARGET):
        p = split.pools[fine_id]
        prev = p.support_for_k(ks[0])
        for k in ks[1:]:
            cur = p.support_for_k(k)
            if not np.array_equal(cur[: prev.size], prev):
                nested = False
            prev = cur
        if not example:
            example = " ⊂ ".join(f"K={k}" for k in ks)
    record("a smaller K_T support is a prefix of a larger one", nested, example)

    # support and query must stay disjoint even at the largest K_T
    worst = True
    for fine_id, p in split.pools.items():
        if p.role != ROLE_TARGET:
            continue
        if np.intersect1d(p.tgt_support_reserve, p.tgt_query).size:
            worst = False
    record("support and query remain disjoint at the largest K_T", worst,
           f"reserve {split.config.tgt_support_reserve}, query takes the remaining {600 - split.config.tgt_support_reserve}")

    # ---- 7. Cross-split leakage ----
    print("\n[4] cross-split leakage")
    sets = {w: set(split.fine_ids(w)) for w in ("train", "val", "test")}
    no_class_leak = (
        not (sets["train"] & sets["test"])
        and not (sets["train"] & sets["val"])
        and not (sets["val"] & sets["test"])
    )
    record("fine classes do not cross splits", no_class_leak,
           f"train {len(sets['train'])} / val {len(sets['val'])} / test {len(sets['test'])}")

    csets = {w: set(split.superclass_split[w]) for w in ("train", "val", "test")}
    no_super_leak = (
        not (csets["train"] & csets["test"])
        and not (csets["train"] & csets["val"])
        and not (csets["val"] & csets["test"])
    )
    record("superclasses do not cross splits", no_super_leak,
           f"train {sorted(csets['train'])} / val {sorted(csets['val'])} / test {sorted(csets['test'])}")

    # No image seen during training may appear in a test episode
    train_imgs = set()
    for w in ("train",):
        for f in split.fine_ids(w):
            p = split.pools[f]
            for arr in (p.src_support, p.src_query, p.tgt_support_reserve, p.tgt_query):
                train_imgs.update(arr.tolist())
    test_imgs = set()
    for f in split.fine_ids("test"):
        p = split.pools[f]
        for arr in (p.src_support, p.src_query, p.tgt_support_reserve, p.tgt_query):
            test_imgs.update(arr.tolist())
    record("zero overlap between training and test images", not (train_imgs & test_imgs),
           f"{len(train_imgs):,} vs {len(test_imgs):,} images")

    # ---- 8. Spot check ----
    print("\n[5] spot check (labels printed for inspection)")
    coarse_id, source_fine, target_fine = split.pairs("test")[0]
    ep = make_episode(split, coarse_id, source_fine, target_fine, k_shot=5)
    pv = ep.provenance
    print(f"  episode: superclass {split.coarse_names[pv.coarse_id]} (id {pv.coarse_id}, split={pv.split})")
    print(f"           source = {split.fine_names[pv.source_fine]} (id {pv.source_fine})")
    print(f"           target = {split.fine_names[pv.target_fine]} (id {pv.target_fine}), K_T={pv.k_shot}")
    for name, idx in ep.all_index_sets().items():
        probe = idx[:6]
        fines = raw.fine[probe]
        coarses = raw.coarse[probe]
        names = {split.fine_names[f] for f in np.unique(raw.fine[idx])}
        print(f"    {name:<12} n={idx.size:<4} first 6 ids={probe.tolist()}")
        print(f"    {'':<12} fine={fines.tolist()} coarse={coarses.tolist()} classes={sorted(names)}")

    # ---- Summary ----
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(RULE)
    print(f"{len(results) - n_fail} / {len(results)} checks passed" + ("" if n_fail == 0 else f"  --  {n_fail} failed"))
    print(RULE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
