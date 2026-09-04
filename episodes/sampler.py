"""Sample Episodes from a MetaSplit.

An episode is uniquely determined by (superclass, source fine class, target fine class, K_T):
all randomness was fixed during build_split and this module only takes slices. The same
arguments therefore always produce the same episode, and no extra seed is needed.

M_S may be reduced at sampling time (the source-scale ablation, item 5 of section 14);
reduction takes a **prefix** of src_support, so the sets stay nested.
"""

from __future__ import annotations

from typing import Iterator

from episodes.guards import check_episode
from episodes.splits import ROLE_SOURCE, ROLE_TARGET, MetaSplit
from episodes.types import Episode, Provenance


def make_episode(
    split: MetaSplit,
    coarse_id: int,
    source_fine: int,
    target_fine: int,
    k_shot: int,
    m_source: int | None = None,
    verify: bool = True,
) -> Episode:
    src = split.pools[source_fine]
    tgt = split.pools[target_fine]

    if src.role != ROLE_SOURCE:
        raise ValueError(f"fine class {source_fine} does not hold the source role")
    if tgt.role != ROLE_TARGET:
        raise ValueError(f"fine class {target_fine} does not hold the target role")
    if src.coarse_id != coarse_id or tgt.coarse_id != coarse_id:
        raise ValueError("source and target must belong to the given superclass")
    if src.split != tgt.split:
        raise ValueError("source and target must belong to the same split")
    if k_shot not in split.config.k_shots:
        raise ValueError(f"K_T={k_shot} is not in {split.config.k_shots}")

    src_support = src.src_support if m_source is None else src.src_support[:m_source]

    ep = Episode(
        task=target_fine,          # the distribution to recover is the target side
        relation=coarse_id,        # the relation is the superclass, not the semantic class
        split=src.split,
        src_support=src_support,
        src_query=src.src_query,
        tgt_support=tgt.support_for_k(k_shot),
        tgt_query=tgt.tgt_query,
        provenance=Provenance(
            split=src.split, coarse_id=coarse_id,
            source_fine=source_fine, target_fine=target_fine,
            k_shot=k_shot, seed=split.config.seed,
        ),
    )
    if verify:
        check_episode(ep)
    return ep


def iter_episodes(
    split: MetaSplit,
    which: str,
    k_shot: int,
    m_source: int | None = None,
    verify: bool = True,
) -> Iterator[Episode]:
    """Iterate every (source, target) combination in a split at a fixed K_T."""
    for coarse_id, s, t in split.pairs(which):
        yield make_episode(split, coarse_id, s, t, k_shot, m_source, verify)


def count_episodes(split: MetaSplit, which: str) -> int:
    return len(split.pairs(which))
