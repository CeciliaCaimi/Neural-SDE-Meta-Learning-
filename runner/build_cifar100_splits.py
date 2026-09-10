"""Build and save a CIFAR-100 meta split, then print a full report.

    python -m runner.build_cifar100_splits                      # sibling scheme
    python -m runner.build_cifar100_splits --scheme domainshift # clean -> corrupted

Output: artifacts/cifar100_split.json, or the path given by --out.

The domain-shift branch used to be missing: `build_domainshift_split` existed in
episodes/domainshift.py but no runner called it, so artifacts/cifar100_domainshift.json
could not be regenerated from the repository. It writes to a **new** file by default
rather than overwriting an existing one, because a split file is the shared input of
every downstream run and silently rebuilding it under a different task family would
invalidate results that are already on disk.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.cifar100 import channel_stats, load_cifar100                # noqa: E402
from domains.corruptions import CORRUPTIONS                               # noqa: E402
from episodes.domainshift import (                                        # noqa: E402
    DomainShiftConfig, build_domainshift_split,
)
from episodes.domainshift import save_split as save_domainshift           # noqa: E402
from episodes.guards import check_split                                   # noqa: E402
from episodes.sampler import count_episodes, iter_episodes                # noqa: E402
from episodes.splits import (                                             # noqa: E402
    ROLE_SOURCE, ROLE_TARGET, SplitConfig, build_split, save_split,
)

OUT = os.path.join(_ROOT, "artifacts", "cifar100_split.json")
OUT_DS = os.path.join(_ROOT, "artifacts", "cifar100_domainshift_c3.json")
RULE = "─" * 78


def build_sibling() -> None:
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
    _report_saved(OUT)


def _sha256(path: str) -> str:
    """The checksum quoted in the protocol card, so three people can verify they
    are reading the same split without comparing files."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fo:
        for chunk in iter(lambda: fo.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_saved(path: str) -> None:
    print()
    print(RULE)
    print(f"saved : {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
    print(f"sha256: {_sha256(path)}")
    print(RULE)


def build_domainshift(corruptions: tuple[str, ...], severity: int, out: str) -> None:
    """The main experiment's split: one semantic class per task, clean source pool,
    corrupted target pool, superclasses divided 12/4/4."""
    cfg = DomainShiftConfig(corruptions=tuple(corruptions), severity=severity)

    print(RULE)
    print("CIFAR-100 DOMAIN-SHIFT SPLIT")
    print(RULE)

    raw = load_cifar100()
    print(f"dataset          : {raw.images.shape[0]} images | {raw.images.shape[1:]} | {raw.images.dtype}")

    split = build_domainshift_split(raw, cfg)

    # The four streams of every class must be pairwise disjoint, and the source pool
    # must never meet the target pool -- otherwise "clean source, corrupted target"
    # degenerates into the clean(X)/corrupt(X) same-image leak the protocol forbids.
    for fid, cs in split.classes.items():
        streams = {
            "src_support": cs.src_support, "src_query": cs.src_query,
            "tgt_support_reserve": cs.tgt_support_reserve, "tgt_query": cs.tgt_query,
        }
        names = list(streams)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = np.intersect1d(streams[a], streams[b])
                if shared.size:
                    raise AssertionError(
                        f"class {fid} ({split.fine_names[fid]}): {a} and {b} share "
                        f"{shared.size} image ids, e.g. {shared[:5].tolist()}"
                    )
        src = np.concatenate([cs.src_support, cs.src_query])
        tgt = np.concatenate([cs.tgt_support_reserve, cs.tgt_query])
        shared = np.intersect1d(src, tgt)
        if shared.size:
            raise AssertionError(f"class {fid}: source and target pools share {shared.size} ids")
    print("stream check     : all four streams pairwise disjoint in every one of "
          f"{len(split.classes)} classes")

    # ---------------- Configuration ----------------
    print()
    print(RULE)
    print("configuration")
    print(RULE)
    print(f"seed                    : {cfg.seed}")
    print(f"superclass split        : train {cfg.n_train_super} / val {cfg.n_val_super} / test {cfg.n_test_super}")
    print(f"source pool (clean)     : support {cfg.src_support_size} + query {cfg.src_query_size} of {cfg.src_pool}")
    print(f"target pool (corrupted) : reserve {cfg.tgt_reserve} + query {cfg.tgt_pool - cfg.tgt_reserve} = {cfg.tgt_pool}")
    print(f"K_T                     : {list(cfg.k_shots)}")
    print(f"relations (corruptions) : {list(cfg.corruptions)}   severity {cfg.severity}")
    print(f"M_S / K_T at K_T=1      : {cfg.src_support_size} / 1   (equation 16 requires M_S >> K_T)")

    # ---------------- Superclass split ----------------
    print()
    print(RULE)
    print("superclass split")
    print(RULE)
    for name in ("train", "val", "test"):
        ids = split.superclass_split[name]
        print(f"\n[{name}]  {len(ids)} superclasses, {split.n_episodes(name)} fine classes")
        for c in ids:
            fines = [f for f in split.fine_ids(name) if split.classes[f].coarse_id == c]
            print(f"  {c:>2}  {split.coarse_names[c]:<32}{', '.join(split.fine_names[f] for f in fines)}")

    # ---------------- Scale ----------------
    print()
    print(RULE)
    print("scale")
    print(RULE)
    hdr = f"{'split':<7}{'sup':>5}{'fine':>6}{'x K_T':>7}{'x cor':>7}{'episode':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("train", "val", "test"):
        n_fine = split.n_episodes(name)
        print(f"{name:<7}{len(split.superclass_split[name]):>5}{n_fine:>6}"
              f"{len(cfg.k_shots):>7}{len(cfg.corruptions):>7}"
              f"{n_fine * len(cfg.k_shots) * len(cfg.corruptions):>9}")

    # ---------------- Shape of a single episode ----------------
    print()
    print(RULE)
    print("the four streams of one episode")
    print(RULE)
    fid = split.fine_ids("train")[0]
    cs = split.classes[fid]
    print(f"{'stream':<22}{'images':>7}   content")
    print("-" * 60)
    print(f"{'src_support':<22}{cs.src_support.size:>7}   {split.fine_names[fid]}, clean")
    print(f"{'src_query':<22}{cs.src_query.size:>7}   {split.fine_names[fid]}, clean")
    print(f"{'tgt_support_reserve':<22}{cs.tgt_support_reserve.size:>7}   {split.fine_names[fid]}, corrupted   <- K_T drawn from here")
    print(f"{'tgt_query':<22}{cs.tgt_query.size:>7}   {split.fine_names[fid]}, corrupted")

    save_domainshift(split, out)
    _report_saved(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a CIFAR-100 meta split")
    ap.add_argument("--scheme", choices=("sibling", "domainshift"), default="sibling")
    ap.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS),
                    help="domainshift only: the relations of the task family")
    ap.add_argument("--severity", type=int, choices=(1, 2, 3, 4, 5), default=3)
    ap.add_argument("--out", default=None,
                    help="output path; defaults to artifacts/cifar100_split.json for the "
                         "sibling scheme and artifacts/cifar100_domainshift_c3.json for "
                         "domainshift. Existing split files are never overwritten silently.")
    a = ap.parse_args()

    if a.scheme == "sibling":
        build_sibling()
        return

    out = a.out or OUT_DS
    if os.path.exists(out):
        raise SystemExit(
            f"{out} already exists. A split file is the shared input of every downstream "
            "run, so this refuses to overwrite one. Pass --out with a new name, or delete "
            "the file deliberately if you mean to replace it."
        )
    build_domainshift(tuple(a.corruptions), a.severity, out)


if __name__ == "__main__":
    main()
