"""Build and save the CIFAR-100 meta split, then print a full report.

    python -m runner.build_cifar100_splits

Output: artifacts/cifar100_split.json
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.cifar100 import channel_stats, load_cifar100                # noqa: E402
from episodes.guards import check_split                                   # noqa: E402
from episodes.sampler import count_episodes, iter_episodes                # noqa: E402
from episodes.splits import (                                             # noqa: E402
    ROLE_SOURCE, ROLE_TARGET, SplitConfig, build_split, save_split,
)

OUT = os.path.join(_ROOT, "artifacts", "cifar100_split.json")
RULE = "─" * 78


def main() -> None:
    cfg = SplitConfig()

    print(RULE)
    print("CIFAR-100 META SPLIT")
    print(RULE)

    raw = load_cifar100()
    mean, std = channel_stats(raw)
    print(f"dataset          : {raw.images.shape[0]} images | {raw.images.shape[1:]} | {raw.images.dtype}")
    print(f"channel mean/std : {np.round(mean, 4).tolist()} / {np.round(std, 4).tolist()}")
    print(f"                   (reference [0.5071 0.4865 0.4409] / [0.2673 0.2564 0.2762])")

    split = build_split(raw, cfg)
    check_split(split)

    # ---------------- Configuration ----------------
    print()
    print(RULE)
    print("configuration")
    print(RULE)
    print(f"seed                    : {cfg.seed}")
    print(f"superclass split        : train {cfg.n_train_super} / val {cfg.n_val_super} / test {cfg.n_test_super}")
    print(f"roles per superclass    : {cfg.n_source_per_super} source | {cfg.n_target_per_super} target")
    print(f"source class pool       : support {cfg.src_support_size} + query {cfg.src_query_size} = {cfg.src_support_size + cfg.src_query_size} / 600")
    print(f"target class pool       : reserve {cfg.tgt_support_reserve} + query {600 - cfg.tgt_support_reserve} = 600 / 600")
    print(f"K_T                     : {list(cfg.k_shots)}  (support nested over K_T)")

    # ---------------- Superclass split ----------------
    print()
    print(RULE)
    print("superclass split")
    print(RULE)
    for name in ("train", "val", "test"):
        ids = split.superclass_split[name]
        if not ids:
            print(f"\n[{name}]  empty -- no superclasses were allocated to this split")
            print("        (the protocol selects hyperparameters on val, so it must be non-empty)")
            continue
        print(f"\n[{name}]  {len(ids)} superclasses")
        for c in ids:
            srcs = [f for f in split.fine_ids(name, ROLE_SOURCE) if split.pools[f].coarse_id == c]
            tgts = [f for f in split.fine_ids(name, ROLE_TARGET) if split.pools[f].coarse_id == c]
            sn = ", ".join(split.fine_names[f] for f in srcs)
            tn = ", ".join(split.fine_names[f] for f in tgts)
            print(f"  {c:>2}  {split.coarse_names[c]:<32}")
            print(f"      source : {sn}")
            print(f"      target : {tn}")

    # ---------------- Scale ----------------
    print()
    print(RULE)
    print("scale")
    print(RULE)
    hdr = f"{'split':<7}{'sup':>5}{'fine':>6}{'src cls':>9}{'tgt cls':>9}{'episode':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("train", "val", "test"):
        print(
            f"{name:<7}{len(split.superclass_split[name]):>5}"
            f"{len(split.fine_ids(name)):>6}"
            f"{len(split.fine_ids(name, ROLE_SOURCE)):>9}"
            f"{len(split.fine_ids(name, ROLE_TARGET)):>9}"
            f"{count_episodes(split, name):>9}"
        )
    total_ep = sum(count_episodes(split, n) for n in ("train", "val", "test"))
    print("-" * len(hdr))
    print(f"{'total':<7}{20:>5}{100:>6}"
          f"{len(split.fine_ids('train', ROLE_SOURCE)) + len(split.fine_ids('val', ROLE_SOURCE)) + len(split.fine_ids('test', ROLE_SOURCE)):>9}"
          f"{len(split.fine_ids('train', ROLE_TARGET)) + len(split.fine_ids('val', ROLE_TARGET)) + len(split.fine_ids('test', ROLE_TARGET)):>9}"
          f"{total_ep:>9}")
    print(f"\n(each episode admits {len(cfg.k_shots)} values of K_T -> {total_ep} x {len(cfg.k_shots)} = {total_ep * len(cfg.k_shots)} (episode, K_T) pairs)")

    # ---------------- Shape of a single episode ----------------
    print()
    print(RULE)
    print("the four streams of one episode")
    print(RULE)
    ep = next(iter_episodes(split, "train", k_shot=5))
    print(f"{'stream':<14}{'images':>7}   drawn from")
    print("-" * 52)
    print(f"{'src_support':<14}{ep.src_support.size:>7}   {split.fine_names[ep.provenance.source_fine]}")
    print(f"{'src_query':<14}{ep.src_query.size:>7}   {split.fine_names[ep.provenance.source_fine]}")
    print(f"{'tgt_support':<14}{ep.tgt_support.size:>7}   {split.fine_names[ep.provenance.target_fine]}   ← K_T")
    print(f"{'tgt_query':<14}{ep.tgt_query.size:>7}   {split.fine_names[ep.provenance.target_fine]}")
    print(f"\nM_S / K_T = {ep.m_source} / {ep.k_shot} = {ep.m_source / ep.k_shot:.0f}x   (equation 16 requires M_S >> K_T)")

    save_split(split, OUT)
    size_kb = os.path.getsize(OUT) / 1024
    print()
    print(RULE)
    print(f"saved: {OUT}  ({size_kb:.0f} KB)")
    print(RULE)


if __name__ == "__main__":
    main()
