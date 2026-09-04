"""Leakage guards.

These assertions are on by default; they **enforce** the protocol rather than describe it:
  - the three semantic-task splits are disjoint; test tasks never enter training or tuning
  - support and query are disjoint within each domain
  - meta-test refinement never touches the target query
  - the transport relation descriptor is indexed by relation, not by semantic class

The last of these is already enforced by the Episode type (relation stores coarse_id);
the rest are checked here.
"""

from __future__ import annotations

import numpy as np

from episodes.splits import ROLE_SOURCE, ROLE_TARGET, SPLIT_NAMES, MetaSplit
from episodes.types import Episode


class LeakageError(AssertionError):
    """The protocol was violated. Do not catch this -- it invalidates the results."""


def _fail(msg: str) -> None:
    raise LeakageError(msg)


def check_split(split: MetaSplit) -> None:
    """Split-file level checks; run once after building a split."""
    cfg = split.config

    # --- the three superclass splits are disjoint and cover all 20 ---
    sets = {k: set(v) for k, v in split.superclass_split.items()}
    if set(sets) != set(SPLIT_NAMES):
        _fail(f"split keys should be {SPLIT_NAMES}, got {sorted(sets)}")
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b and sets[a] & sets[b]:
                _fail(f"a superclass appears in both {a} and {b}: {sorted(sets[a] & sets[b])}")
    union = set().union(*sets.values())
    if len(union) != 20:
        _fail(f"there should be 20 superclasses in total, got {len(union)}")
    if len(sets["train"]) != cfg.n_train_super:
        _fail(f"train should hold {cfg.n_train_super} superclasses, got {len(sets['train'])}")

    # --- fine classes do not cross splits, and role assignment matches the config ---
    per_super: dict[int, dict[str, list[int]]] = {}
    for fine_id, p in split.pools.items():
        if split.split_of_coarse(p.coarse_id) != p.split:
            _fail(f"fine class {fine_id} has split={p.split}, inconsistent with superclass {p.coarse_id}")
        per_super.setdefault(p.coarse_id, {ROLE_SOURCE: [], ROLE_TARGET: []})
        per_super[p.coarse_id][p.role].append(fine_id)

    for coarse_id, roles in per_super.items():
        if len(roles[ROLE_SOURCE]) != cfg.n_source_per_super:
            _fail(f"superclass {coarse_id} has {len(roles[ROLE_SOURCE])} source fine classes")
        if len(roles[ROLE_TARGET]) != cfg.n_target_per_super:
            _fail(f"superclass {coarse_id} has {len(roles[ROLE_TARGET])} target fine classes")
        if set(roles[ROLE_SOURCE]) & set(roles[ROLE_TARGET]):
            _fail(f"superclass {coarse_id} has a fine class that is both source and target")

    # --- within each fine class, the pools are pairwise disjoint ---
    for fine_id, p in split.pools.items():
        pools = {
            "src_support": p.src_support, "src_query": p.src_query,
            "tgt_support_reserve": p.tgt_support_reserve, "tgt_query": p.tgt_query,
        }
        _assert_pairwise_disjoint(pools, f"fine class {fine_id}")
        used = np.concatenate([v for v in pools.values() if v.size])
        if np.unique(used).size != used.size:
            _fail(f"fine class {fine_id} has duplicate ids inside its image pool")

    # --- global: no image may belong to two fine classes (impossible by construction, but cheap) ---
    everything = np.concatenate(
        [v for p in split.pools.values()
         for v in (p.src_support, p.src_query, p.tgt_support_reserve, p.tgt_query)
         if v.size]
    )
    if np.unique(everything).size != everything.size:
        _fail("the same image was assigned to more than one fine class")


def check_episode(ep: Episode) -> None:
    """Every sampled episode must pass this."""
    _assert_pairwise_disjoint(ep.all_index_sets(), f"episode {ep.provenance}")

    if ep.provenance.source_fine == ep.provenance.target_fine:
        _fail("source and target are the same fine class")
    if ep.relation != ep.provenance.coarse_id:
        _fail("relation must equal the superclass id (indexed by relation, not by semantic class)")
    if ep.k_shot != ep.provenance.k_shot:
        _fail(f"inconsistent K_T: {ep.k_shot} vs {ep.provenance.k_shot}")

    # The defining condition M_S >> K_T (equation 16)
    if ep.m_source <= ep.k_shot:
        _fail(f"M_S >> K_T violated: M_S={ep.m_source}, K_T={ep.k_shot}")


def check_episode_labels(ep: Episode, raw) -> None:
    """Loaded images must carry the label of the class they were assigned to.

    This is the check required by rule 3: it stops an index mismatch from quietly ruining a run.
    """
    pv = ep.provenance
    for name in ("src_support", "src_query"):
        idx = ep.all_index_sets()[name]
        bad = idx[raw.fine[idx] != pv.source_fine]
        if bad.size:
            _fail(f"{name}: {bad.size} images do not carry fine label {pv.source_fine}")
    for name in ("tgt_support", "tgt_query"):
        idx = ep.all_index_sets()[name]
        bad = idx[raw.fine[idx] != pv.target_fine]
        if bad.size:
            _fail(f"{name}: {bad.size} images do not carry fine label {pv.target_fine}")
    # Superclass consistency: all four streams must belong to this superclass
    for name, idx in ep.all_index_sets().items():
        bad = idx[raw.coarse[idx] != pv.coarse_id]
        if bad.size:
            _fail(f"{name}: {bad.size} images do not carry coarse label {pv.coarse_id}")


def _assert_pairwise_disjoint(named: dict[str, np.ndarray], ctx: str) -> None:
    keys = [k for k, v in named.items() if v.size]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = np.intersect1d(named[a], named[b], assume_unique=False)
            if overlap.size:
                _fail(f"{ctx}: {a} and {b} overlap in {overlap.size} images (e.g. {overlap[:5].tolist()})")
