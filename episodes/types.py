"""Episode -- the carrier of the protocol, matching the tuple of A.1:

    e_y = (y, d_S, d_T, D^s_{y,S}, D^q_{y,S}, D^s_{y,T}, D^q_{y,T}, c_{S->T})

What is stored here are **global image ids**, not pixels. Pixels are fetched on
demand in materialise(), which keeps an episode down to a few kilobytes so that
whole batches of them can be written into the split file for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domains.cifar100 import CIFAR100Raw


@dataclass(frozen=True)
class Provenance:
    """The minimum information needed to reconstruct any episode exactly."""

    split: str              # train | val | test
    coarse_id: int          # superclass: the scope within which the relation acts
    source_fine: int        # fine class taking the source role
    target_fine: int        # fine class taking the target role
    k_shot: int
    seed: int


@dataclass(frozen=True)
class Episode:
    task: int               # y -- here the target_fine, the distribution to recover
    relation: int           # c_{S->T} -- indexed by relation, never by semantic class
    split: str

    src_support: np.ndarray   # D^s_{y,S}
    src_query: np.ndarray     # D^q_{y,S}
    tgt_support: np.ndarray   # D^s_{y,T},  size K_T
    tgt_query: np.ndarray     # D^q_{y,T},  evaluation only

    provenance: Provenance

    @property
    def m_source(self) -> int:
        return int(self.src_support.size)

    @property
    def k_shot(self) -> int:
        return int(self.tgt_support.size)

    def all_index_sets(self) -> dict[str, np.ndarray]:
        return {
            "src_support": self.src_support,
            "src_query": self.src_query,
            "tgt_support": self.tgt_support,
            "tgt_query": self.tgt_query,
        }

    def materialise(self, raw: CIFAR100Raw, which: str) -> np.ndarray:
        """Fetch the pixels of one stream as (N, 32, 32, 3) uint8."""
        return raw.images[self.all_index_sets()[which]]
