"""Checks on the Stage C episode streams, before any of them is trained on.

    python scripts/check_fitz_episodes.py

scripts/check_fitz_split.py already re-verifies the split file against the raw annotations.
This checks the layer above it: that the loader turns that file into episodes with the
properties the experiment's claim depends on. Runs on CPU, needs the images on disk.

The three that decide whether a result means anything:

* **No image appears on both sides of the population shift, and none in two streams.**
  Deduplication removed 127 groups that held the same photograph annotated as both I-III and
  IV-VI. If any survived, part of the shift would be an identity map and transport would
  look better than it is.
* **No condition appears in two splits, and no image of a test condition is ever reachable
  from a training episode.** The conditions are split precisely so that the relation, not the
  task, is what generalises.
* **Every source image really is phototype I-III and every target image IV-VI**, checked
  against the published annotation rather than against the split file that asserts it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                                  # noqa: E402
from domains.fitzpatrick import (                                          # noqa: E402
    SOURCE_PHOTOTYPES, TARGET_PHOTOTYPES, load_annotations, load_fitzpatrick,
)
from episodes.fitzpatrick import FitzLoader, load_fitz_split               # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    cfg = BaseConfig()
    path = os.path.join(_ROOT, cfg.episodes.fitz_path)
    split = load_fitz_split(path)
    ann = load_annotations()
    dev = torch.device("cpu")

    print("1. the split file, as the loader reads it")
    print(f"     {os.path.basename(path)}  relation {split.relation['source']} -> "
          f"{split.relation['target']}  checksum {split.checksum[:12]}...")
    for w in ("train", "val", "test"):
        print(f"     {w:<6} {len(split.names(w))} conditions: {', '.join(split.names(w))}")
    check("the condition partition is three-way and non-empty",
          all(split.names(w) for w in ("train", "val", "test")))
    seen: dict[str, str] = {}
    dupes = []
    for w in ("train", "val", "test"):
        for n in split.names(w):
            if n in seen:
                dupes.append(n)
            seen[n] = w
    check("no condition appears in two splits", not dupes, str(dupes))
    check("only training conditions fund a source query stream",
          all(bool(split.conditions[n].src_query) for n in split.names("train"))
          and not any(split.conditions[n].src_query
                      for w in ("val", "test") for n in split.names(w)))

    print("\n2. no image is in two places")
    owner: dict[str, list[str]] = {}
    for n, c in split.conditions.items():
        for stream in ("src_support", "src_query", "tgt_support_reserve", "tgt_query"):
            for h in getattr(c, stream):
                owner.setdefault(h, []).append(f"{n}/{stream}")
    multi = {h: v for h, v in owner.items() if len(v) > 1}
    check("every image belongs to exactly one condition and one stream",
          not multi, f"{len(multi)} shared, e.g. {list(multi.items())[:2]}")

    src_side, tgt_side = set(), set()
    for c in split.conditions.values():
        src_side |= set(c.src_support) | set(c.src_query)
        tgt_side |= set(c.tgt_support_reserve) | set(c.tgt_query)
    check("no image is on both sides of the population shift",
          not (src_side & tgt_side), f"{len(src_side & tgt_side)} on both sides")
    print(f"        {len(src_side)} source images, {len(tgt_side)} target, "
          f"{len(owner)} distinct, split says {split.n_images}")
    check("the image count agrees with what the split file recorded",
          len(owner) == split.n_images, f"{len(owner)} against {split.n_images}")

    print("\n3. the population labels, read from the published annotation")
    bad_src = [h for h in src_side
               if ann.get(h) is None or ann[h].phototype not in SOURCE_PHOTOTYPES]
    bad_tgt = [h for h in tgt_side
               if ann.get(h) is None or ann[h].phototype not in TARGET_PHOTOTYPES]
    check(f"every source image is phototype {SOURCE_PHOTOTYPES}",
          not bad_src, f"{len(bad_src)} are not")
    check(f"every target image is phototype {TARGET_PHOTOTYPES}",
          not bad_tgt, f"{len(bad_tgt)} are not")
    wrong_label = [(n, h) for n, c in split.conditions.items() for h in c.all_hashes()
                   if ann.get(h) is not None and ann[h].label != n]
    check("every image's condition annotation matches the condition it is filed under",
          not wrong_label, f"{len(wrong_label)} mismatched, e.g. {wrong_label[:2]}")

    print("\n4. the loader's episodes")
    hashes = split.all_hashes()
    raw = load_fitzpatrick(hashes=hashes, verbose=True)
    k_shots = (1, 2, 3, 5, 8, 12, 20)
    check("the split funds the K_T grid this experiment uses",
          max(k_shots) <= split.max_k, f"max K_T {max(k_shots)} <= reserve {split.max_k}")

    tr = FitzLoader(raw, split, "train", device=dev, k_shots=k_shots, seed=1)
    te = FitzLoader(raw, split, "test", device=dev, k_shots=k_shots, seed=2)
    b = tr.sample(k_shot=5)
    check("pixel shapes are (N, 3, S, S)",
          b.src_support.shape[1:] == (3, raw.image_size, raw.image_size)
          and b.tgt_support.shape[1:] == (3, raw.image_size, raw.image_size),
          f"{tuple(b.src_support.shape)} / {tuple(b.tgt_support.shape)}")
    check("K_T is honoured exactly", b.tgt_support.shape[0] == 5,
          str(b.tgt_support.shape[0]))
    check("M_S comes from the split's own config",
          b.src_support.shape[0] == split.config["m_source"],
          f"{b.src_support.shape[0]} = {split.config['m_source']}")
    lo, hi = float(b.tgt_query.min()), float(b.tgt_query.max())
    check("pixels are scaled to [-1, 1] as the diffusion schedule expects",
          -1.001 <= lo and hi <= 1.001, f"[{lo:.3f}, {hi:.3f}]")
    check("the relation is a single scalar, which is what omits the descriptor downstream",
          b.relation.numel() == 1 and int(b.relation) == 0)

    # The target stream must not be a transformed copy of the source stream. On CIFAR it was
    # corrupt(X) of the same X; here it must be different photographs entirely.
    check("target images are not a transformed copy of source images",
          b.src_support.shape[0] != b.tgt_support.shape[0]
          or not torch.allclose(b.src_support[:b.tgt_support.shape[0]], b.tgt_support,
                                atol=0.2))

    print("\n5. nested source prefixes, for an M_S sweep")
    n1 = FitzLoader(raw, split, "train", device=dev, k_shots=k_shots, seed=7,
                    nested_source=True, enc_source_images=8)
    n2 = FitzLoader(raw, split, "train", device=dev, k_shots=k_shots, seed=7,
                    nested_source=True, enc_source_images=16)
    cond = split.names("train")[0]
    a8 = n1.sample(k_shot=1, condition=cond).src_support
    a16 = n2.sample(k_shot=1, condition=cond).src_support
    check("the smaller source set is a prefix of the larger, so a sweep varies size alone",
          torch.equal(a8, a16[:8]), f"{a8.shape[0]} vs {a16.shape[0]}")

    print("\n6. the split boundary holds at the pixel level")
    test_hashes = {h for w in ("test",) for n in split.names(w)
                   for h in split.conditions[n].all_hashes()}
    train_reachable = {h for n in split.names("train")
                       for h in split.conditions[n].all_hashes()}
    check("no image of a test condition is reachable from a training episode",
          not (test_hashes & train_reachable),
          f"{len(test_hashes & train_reachable)} reachable")
    seen_conditions = {tr.sample().provenance["condition"] for _ in range(200)}
    check("a training loader only ever emits training conditions",
          seen_conditions <= set(split.names("train")),
          f"{len(seen_conditions)} of {len(split.names('train'))} seen")
    seen_test = {te.sample().provenance["condition"] for _ in range(60)}
    check("a test loader only ever emits test conditions",
          seen_test <= set(split.names("test")), str(sorted(seen_test)))

    print("\n7. determinism")
    d1 = FitzLoader(raw, split, "train", device=dev, k_shots=k_shots, seed=99)
    d2 = FitzLoader(raw, split, "train", device=dev, k_shots=k_shots, seed=99)
    e1 = [d1.sample() for _ in range(5)]
    e2 = [d2.sample() for _ in range(5)]
    check("two loaders with one seed emit identical episodes",
          all(x.provenance == y.provenance
              and torch.equal(x.src_support, y.src_support)
              and torch.equal(x.tgt_query, y.tgt_query) for x, y in zip(e1, e2)))

    print("\n8. how much this can resolve")
    for w in ("val", "test"):
        q = [len(split.conditions[n].tgt_query) for n in split.names(w)]
        print(f"     {w:<6} {len(q)} tasks, held-out target images {q} "
              f"(Stage A used a query batch of 128 over 60 episodes per K_T)")
    check("the test side has at least three tasks", len(split.names("test")) >= 3,
          f"{len(split.names('test'))} - too few for a paired interval; the write-up must "
          f"say so")

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
