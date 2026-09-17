"""Stage C episodes: one skin condition is one task, the phototype shift is the relation.

The split file is built by runner/build_fitz_splits.py and is read, never rebuilt, here. It
stores md5 hashes per stream rather than positions, so it survives a change in which images
are on disk, and it refuses to be overwritten.

    e = (condition, D^s_{y,S}, D^q_{y,S}, D^s_{y,T} of size K_T, D^q_{y,T})

Four things differ from episodes/domainshift.py and each is a consequence of the data, not a
preference:

1. **The target stream is not transformed.** Target images are different photographs of
   different patients whose phototype is IV-VI. Nothing is corrupted, so the leak that
   clean(X) / corrupt(X) pairs would create cannot arise -- and the deduplication that
   removed 127 groups holding the same photograph on both sides is what makes that true.
2. **There is one relation.** Phototype I-III to IV-VI is a single relation, so
   training/loop.py leaves n_relations at None, the relation descriptor is omitted and
   transport.relation_emb is None. The relation-only control therefore degenerates to one
   constant and stops being distinguishable from the mean-source control: A1's panel cannot
   be copied across unchanged.
3. **The conditions themselves are split** train/validation/test, which is what makes the
   relation rather than the task the object of generalisation. A model is never evaluated on
   a condition it trained on.
4. **Only training conditions carry a source query stream.** src_query feeds the source
   denoising term of the meta objective and nothing at meta-test reads it, so the split
   spends no images on it for validation and test conditions -- which is what lets those
   conditions exist at all on the reachable archive.

The class mirrors DomainShiftLoader's interface (sample, sample_many, images, fids) so the
training loop, the diagnostics and every evaluation script reach it unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from domains.fitzpatrick import FitzRaw


@dataclass(frozen=True)
class ConditionStreams:
    name: str
    split: str
    src_support: list[str] = field(default_factory=list)
    src_query: list[str] = field(default_factory=list)
    tgt_support_reserve: list[str] = field(default_factory=list)
    tgt_query: list[str] = field(default_factory=list)

    def all_hashes(self) -> list[str]:
        return [*self.src_support, *self.src_query,
                *self.tgt_support_reserve, *self.tgt_query]


@dataclass(frozen=True)
class FitzSplit:
    relation: dict
    config: dict
    conditions: dict[str, ConditionStreams]
    dedup: dict
    n_images: int
    checksum: str
    path: str

    def names(self, split: str) -> list[str]:
        return sorted(n for n, c in self.conditions.items() if c.split == split)

    def all_hashes(self, splits: tuple[str, ...] = ("train", "val", "test")) -> list[str]:
        """Every hash any stream of the named splits will ask for. Pass this to
        load_fitzpatrick so decoding touches no file the run does not read."""
        out: list[str] = []
        for c in self.conditions.values():
            if c.split in splits:
                out += c.all_hashes()
        return sorted(set(out))

    @property
    def max_k(self) -> int:
        """The support reserve is a nested prefix, so this is the largest K_T the split
        can fund. Extra values below it cost compute and no images."""
        return min(len(c.tgt_support_reserve) for c in self.conditions.values())


def load_fitz_split(path: str) -> FitzSplit:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("dataset") != "fitzpatrick17k":
        raise ValueError(f"{path} is not a Fitzpatrick split (dataset={d.get('dataset')!r})")
    conds = {
        name: ConditionStreams(
            name=name, split=c["split"],
            src_support=list(c["src_support"]), src_query=list(c["src_query"]),
            tgt_support_reserve=list(c["tgt_support_reserve"]),
            tgt_query=list(c["tgt_query"]),
        )
        for name, c in d["conditions"].items()
    }
    return FitzSplit(relation=d["relation"], config=d["config"], conditions=conds,
                     dedup=d.get("dedup", {}), n_images=d.get("n_images", 0),
                     checksum=d.get("checksum", ""), path=path)


@dataclass(frozen=True)
class FitzBatch:
    src_support: Tensor
    src_query: Tensor
    tgt_support: Tensor
    tgt_query: Tensor
    relation: Tensor
    provenance: dict


class FitzLoader:
    """Samples episodes from one split of the condition partition.

    ``query_batch`` is capped per condition by what that condition holds: the test side
    carries 57, 34 and 31 held-out target images against Stage A's 128, so a distance
    computed here is materially noisier than anything in Stage A and any comparison across
    the two stages has to say so.
    """

    def __init__(self, raw: FitzRaw, split: FitzSplit, which: str,
                 device="cuda", enc_source_images: int | None = None, query_batch: int = 32,
                 k_shots=None, seed: int = 0, nested_source: bool = False):
        self.split = split
        self.which = which
        self.device = torch.device(device)
        self.enc_source_images = int(enc_source_images or split.config["m_source"])
        self.query_batch = int(query_batch)
        self.nested_source = bool(nested_source)
        self.k_shots = tuple(k_shots or (1, 2, 3, 5, 8, 12, 20))
        if max(self.k_shots) > split.max_k:
            raise ValueError(
                f"k_shots asks for K_T={max(self.k_shots)} but the split reserves only "
                f"{split.max_k} target support images per condition")
        self.fids = split.names(which)                  # named fids to mirror the CIFAR loader
        if not self.fids:
            raise ValueError(f"split '{which}' contains no conditions")
        self.rng = np.random.default_rng(seed)
        self.raw = raw
        # (N, S, S, 3) uint8 resident on the device, fetched by index like the CIFAR loader
        self.images = torch.from_numpy(np.ascontiguousarray(raw.images)).to(self.device)

    # ---- pixels -----------------------------------------------------------

    def _fetch(self, hashes: list[str]) -> Tensor:
        gi = torch.from_numpy(self.raw.indices(list(hashes))).to(self.device).long()
        return self.images.index_select(0, gi).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)

    def _sub(self, pool: list[str], n: int) -> list[str]:
        if n >= len(pool):
            return list(pool)
        sel = self.rng.choice(len(pool), n, replace=False)
        return [pool[int(i)] for i in sel]

    def _sub_source(self, pool: list[str], n: int) -> list[str]:
        """Nested prefixes when a sweep varies M_S, independent draws otherwise -- the same
        distinction, and the same reason, as episodes/domainshift.py:_sub_source."""
        if not self.nested_source:
            return self._sub(pool, n)
        return list(pool) if n >= len(pool) else list(pool[:n])

    # ---- episodes ---------------------------------------------------------

    def sample(self, k_shot: int | None = None, condition: str | None = None) -> FitzBatch:
        name = condition or str(self.fids[self.rng.integers(len(self.fids))])
        cs = self.split.conditions[name]
        k = int(k_shot if k_shot is not None else self.rng.choice(self.k_shots))
        if k > len(cs.tgt_support_reserve):
            raise ValueError(f"{name} reserves {len(cs.tgt_support_reserve)} target support "
                             f"images, K_T={k} asked for")
        if not cs.src_query and self.which == "train":
            raise ValueError(
                f"condition '{name}' is in the training split but has no source query "
                "stream; the meta objective's source term cannot be formed")
        # Validation and test conditions fund no source query stream. Nothing at meta-test
        # reads it, so the source query falls back to the support pool rather than failing:
        # a diagnostic run over val episodes needs *some* source query batch.
        src_q_pool = cs.src_query or cs.src_support
        return FitzBatch(
            src_support=self._fetch(self._sub_source(cs.src_support, self.enc_source_images)),
            src_query=self._fetch(self._sub(src_q_pool, self.query_batch)),
            tgt_support=self._fetch(cs.tgt_support_reserve[:k]),
            tgt_query=self._fetch(self._sub(cs.tgt_query, self.query_batch)),
            # one relation, so the descriptor is omitted downstream; kept for interface parity
            relation=torch.tensor(0, device=self.device, dtype=torch.long),
            provenance={"condition": name, "k": k, "split": self.which,
                        "src_query_from_support": not cs.src_query},
        )

    def sample_many(self, n: int, k_shot: int | None = None) -> list[FitzBatch]:
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.fids) * len(self.k_shots)
