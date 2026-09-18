r"""Stage C episodes, second attempt: a thoracic finding is the task, an age shift the relation.

    e = (finding, relation, D^s_{y,S}, D^q_{y,S}, D^s_{y,T} of size K_T, D^q_{y,T})

The one structural difference from episodes/fitzpatrick.py is the one that matters. There,
phototype I-III to IV-VI was a single relation, so `n_relations` stayed None, the relation
descriptor was omitted, and C4 found the consequence: the same shift applies to every episode,
a constant coordinate solves the meta-objective, the encoder's conditions ended up 1.4 % of a
coordinate norm apart, and knowing the correct source task was worth nothing.

Here one source age bin faces **three** target bins, so an episode is (finding, relation) and
the relation descriptor is carried, exactly as the corruption type is on CIFAR. That restores
the structure the method was shown to work in, and it is what makes the source-dependence
controls meaningful again: a within-relation shuffled source is now a different thing from a
relation-only coordinate.

The source stream is shared across a task's relations, because the source population is the
same age bin for all three. The target streams are per relation.

Everything else follows episodes/domainshift.py's interface -- sample, sample_many, images,
fids -- so the training loop, the diagnostics and every evaluation script reach it unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from domains.chestxray import CXRRaw


@dataclass(frozen=True)
class TaskStreams:
    name: str
    split: str
    src_support: list[str] = field(default_factory=list)
    src_query: list[str] = field(default_factory=list)
    # relation name -> {"bin": int, "tgt_support_reserve": [...], "tgt_query": [...]}
    targets: dict[str, dict] = field(default_factory=dict)

    def all_names(self) -> list[str]:
        out = [*self.src_support, *self.src_query]
        for t in self.targets.values():
            out += t["tgt_support_reserve"] + t["tgt_query"]
        return out


@dataclass(frozen=True)
class CXRSplit:
    source: dict
    relations: list[dict]
    config: dict
    tasks: dict[str, TaskStreams]
    checksum: str
    path: str

    def names(self, split: str) -> list[str]:
        return sorted(n for n, t in self.tasks.items() if t.split == split)

    @property
    def relation_names(self) -> list[str]:
        return [r["name"] for r in self.relations]

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    def all_images(self, splits: tuple[str, ...] = ("train", "val", "test")) -> list[str]:
        out: list[str] = []
        for t in self.tasks.values():
            if t.split in splits:
                out += t.all_names()
        return sorted(set(out))

    @property
    def max_k(self) -> int:
        return min(len(tt["tgt_support_reserve"])
                   for t in self.tasks.values() for tt in t.targets.values())


def load_cxr_split(path: str) -> CXRSplit:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("dataset") != "chestxray14":
        raise ValueError(f"{path} is not a ChestX-ray14 split (dataset={d.get('dataset')!r})")
    tasks = {n: TaskStreams(name=n, split=t["split"],
                            src_support=list(t["src_support"]),
                            src_query=list(t["src_query"]),
                            targets={k: dict(v) for k, v in t["targets"].items()})
             for n, t in d["tasks"].items()}
    return CXRSplit(source=d["source"], relations=d["relations"], config=d["config"],
                    tasks=tasks, checksum=d.get("checksum", ""), path=path)


@dataclass(frozen=True)
class CXRBatch:
    src_support: Tensor
    src_query: Tensor
    tgt_support: Tensor
    tgt_query: Tensor
    relation: Tensor
    provenance: dict


class CXRLoader:
    """Samples episodes from one split of the task partition."""

    def __init__(self, raw: CXRRaw, split: CXRSplit, which: str,
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
                f"{split.max_k} target support images per (task, relation)")
        self.fids = split.names(which)
        if not self.fids:
            raise ValueError(f"split '{which}' contains no tasks")
        self.relations = split.relation_names
        self.rng = np.random.default_rng(seed)
        self.raw = raw
        self.images = torch.from_numpy(np.ascontiguousarray(raw.images)).to(self.device)

    # ---- pixels -----------------------------------------------------------

    def _fetch(self, names: list[str]) -> Tensor:
        gi = torch.from_numpy(self.raw.indices(list(names))).to(self.device).long()
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

    def sample(self, k_shot: int | None = None, task: str | None = None,
               relation: str | None = None) -> CXRBatch:
        name = task or str(self.fids[self.rng.integers(len(self.fids))])
        ts = self.split.tasks[name]
        rel = relation or str(self.relations[self.rng.integers(len(self.relations))])
        if rel not in ts.targets:
            raise ValueError(f"task '{name}' has no relation '{rel}'")
        k = int(k_shot if k_shot is not None else self.rng.choice(self.k_shots))
        tt = ts.targets[rel]
        if k > len(tt["tgt_support_reserve"]):
            raise ValueError(f"{name}/{rel} reserves {len(tt['tgt_support_reserve'])} target "
                             f"support images, K_T={k} asked for")
        if not ts.src_query and self.which == "train":
            raise ValueError(
                f"task '{name}' is in the training split but has no source query stream")
        # Validation and test tasks fund no source query stream -- nothing at meta-test reads
        # it -- so a diagnostic pass over them falls back to the support pool rather than
        # failing outright.
        src_q_pool = ts.src_query or ts.src_support
        return CXRBatch(
            src_support=self._fetch(self._sub_source(ts.src_support, self.enc_source_images)),
            src_query=self._fetch(self._sub(src_q_pool, self.query_batch)),
            tgt_support=self._fetch(tt["tgt_support_reserve"][:k]),
            tgt_query=self._fetch(self._sub(tt["tgt_query"], self.query_batch)),
            relation=torch.tensor(self.relations.index(rel), device=self.device,
                                  dtype=torch.long),
            provenance={"task": name, "relation": rel, "k": k, "split": self.which,
                        "src_query_from_support": not ts.src_query},
        )

    def sample_many(self, n: int, k_shot: int | None = None) -> list[CXRBatch]:
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.fids) * len(self.relations) * len(self.k_shots)
