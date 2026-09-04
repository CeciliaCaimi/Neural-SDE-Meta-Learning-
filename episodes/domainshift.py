"""clean -> corrupted domain-shift split and loader (sections 3.3 and 12.3).

This sits alongside the sibling fine-class scheme (episodes/splits.py) without affecting it:

  - one semantic task = one fine class (no source-role / target-role division)
  - source = clean images of that class; target = other images of the same class, corrupted
  - the relation c_{S->T} is the corruption type; c is omitted for a single fixed corruption
  - classes are split by superclass into train/val/test (12/4/4 -> 60/20/20 fine classes)
  - protocol: hyperparameters and budgets are chosen on val only; test is for final reporting

Each class's 600 images are divided into two disjoint pools: the source pool stays clean and
the target pool is corrupted. Source and target therefore use **different underlying images**,
faithful to real domain shift and free of the clean(X) / corrupt(X) same-image leak.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import Tensor

from domains.cifar100 import IMAGES_PER_FINE, N_COARSE, CIFAR100Raw
from domains.corruptions import CORRUPTION_ID, apply_corruption


@dataclass(frozen=True)
class DomainShiftConfig:
    seed: int = 12345
    n_train_super: int = 12
    n_val_super: int = 4         # hyperparameters and budgets are selected on val (protocol)
    n_test_super: int = 4
    src_pool: int = 350          # the first 350 images of each class are source (kept clean)
    src_support_size: int = 300  # supports M_S in {16, 32, 64, 128, 256, all=300}
    src_query_size: int = 50
    # target pool = 600 - src_pool = 250: reserve max(k) for support, query takes the rest
    k_shots: tuple[int, ...] = (1, 2, 5, 10, 20)
    # Main experiment = one fixed semantics-preserving transformation (protocol: learn the same
    # relation on the training classes). Several corruptions become several relations: a control.
    corruptions: tuple[str, ...] = ("gaussian_blur",)
    severity: int = 3

    def __post_init__(self):
        if self.n_train_super + self.n_val_super + self.n_test_super != N_COARSE:
            raise ValueError("superclass allocation must sum to 20")
        if self.src_support_size + self.src_query_size > self.src_pool:
            raise ValueError("source support + query exceeds the source pool")
        if max(self.k_shots) + 1 > IMAGES_PER_FINE - self.src_pool:
            raise ValueError("the target pool cannot hold the support reserve")

    @property
    def tgt_pool(self) -> int:
        return IMAGES_PER_FINE - self.src_pool

    @property
    def tgt_reserve(self) -> int:
        return max(self.k_shots)


@dataclass(frozen=True)
class ClassStreams:
    fine_id: int
    coarse_id: int
    split: str
    src_support: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    src_query: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    tgt_support_reserve: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    tgt_query: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))


@dataclass(frozen=True)
class DomainShiftSplit:
    config: DomainShiftConfig
    superclass_split: dict[str, list[int]]
    classes: dict[int, ClassStreams]
    fine_names: tuple[str, ...]
    coarse_names: tuple[str, ...]

    def fine_ids(self, split: str) -> list[int]:
        return sorted(f for f, c in self.classes.items() if c.split == split)

    def n_episodes(self, split: str) -> int:
        return len(self.fine_ids(split))


def _rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def build_domainshift_split(raw: CIFAR100Raw, cfg: DomainShiftConfig | None = None) -> DomainShiftSplit:
    cfg = cfg or DomainShiftConfig()
    order = _rng(cfg.seed, 0).permutation(N_COARSE)
    a, b = cfg.n_train_super, cfg.n_train_super + cfg.n_val_super
    superclass_split = {
        "train": sorted(int(c) for c in order[:a]),
        "val":   sorted(int(c) for c in order[a:b]),
        "test":  sorted(int(c) for c in order[b:]),
    }
    coarse_of_split = {}
    for w in ("train", "val", "test"):
        coarse_of_split.update({c: w for c in superclass_split[w]})

    classes: dict[int, ClassStreams] = {}
    for fine_id in range(100):
        coarse_id = int(raw.fine_to_coarse[fine_id])
        split = coarse_of_split[coarse_id]
        idx = _rng(cfg.seed, 2000 + fine_id).permutation(
            raw.indices_of_fine(fine_id)).astype(np.int32)
        src, tgt = idx[:cfg.src_pool], idx[cfg.src_pool:]
        a, b = cfg.src_support_size, cfg.src_support_size + cfg.src_query_size
        classes[fine_id] = ClassStreams(
            fine_id=fine_id, coarse_id=coarse_id, split=split,
            src_support=src[:a], src_query=src[a:b],
            tgt_support_reserve=tgt[:cfg.tgt_reserve], tgt_query=tgt[cfg.tgt_reserve:],
        )
    return DomainShiftSplit(cfg, superclass_split, classes,
                            raw.fine_names, raw.coarse_names)


# --------------------------------------------------------------------------
# Loading: the target streams are corrupted as their pixels are fetched
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DSBatch:
    src_support: Tensor
    src_query: Tensor
    tgt_support: Tensor
    tgt_query: Tensor
    relation: Tensor
    provenance: dict


class DomainShiftLoader:
    def __init__(self, raw, split: DomainShiftSplit, which: str,
                 device="cuda", enc_source_images=64, query_batch=32,
                 k_shots=None, seed=0):
        self.split = split
        self.which = which
        self.device = torch.device(device)
        self.enc_source_images = enc_source_images
        self.query_batch = query_batch
        self.k_shots = tuple(k_shots or split.config.k_shots)
        self.corruptions = tuple(split.config.corruptions)
        self.severity = split.config.severity
        self.fids = split.fine_ids(which)
        if not self.fids:
            raise ValueError(f"split '{which}' contains no classes")
        self.rng = np.random.default_rng(seed)
        imgs = torch.from_numpy(np.ascontiguousarray(raw.images))
        self.images = imgs.to(self.device)

    def _fetch(self, idx: np.ndarray, corruption: str | None) -> Tensor:
        gi = torch.from_numpy(np.ascontiguousarray(idx)).to(self.device).long()
        x = self.images.index_select(0, gi).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        if corruption is not None:
            gids = torch.from_numpy(np.ascontiguousarray(idx)).to(self.device)
            x = apply_corruption(x, corruption, self.severity, gids)
        return x

    def _sub(self, pool: np.ndarray, n: int) -> np.ndarray:
        return pool if n >= pool.size else self.rng.choice(pool, n, replace=False)

    def sample(self, k_shot: int | None = None,
               corruption: str | None = None) -> DSBatch:
        """One episode = (semantic class, corruption, K_T). The corruption is its S->T relation."""
        fid = int(self.fids[self.rng.integers(len(self.fids))])
        cs = self.split.classes[fid]
        k = int(k_shot if k_shot is not None else self.rng.choice(self.k_shots))
        cor = corruption or str(self.rng.choice(self.corruptions))
        return DSBatch(
            src_support=self._fetch(self._sub(cs.src_support, self.enc_source_images), None),
            src_query=self._fetch(self._sub(cs.src_query, self.query_batch), None),
            tgt_support=self._fetch(cs.tgt_support_reserve[:k], cor),
            tgt_query=self._fetch(self._sub(cs.tgt_query, self.query_batch), cor),
            relation=torch.tensor(CORRUPTION_ID[cor], device=self.device, dtype=torch.long),
            provenance={"fine": fid, "corruption": cor, "k": k},
        )

    def sample_many(self, n: int, k_shot: int | None = None) -> list[DSBatch]:
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.fids) * len(self.k_shots) * len(self.corruptions)


def save_split(split: DomainShiftSplit, path: str) -> None:
    payload = {
        "config": asdict(split.config),
        "superclass_split": split.superclass_split,
        "fine_names": list(split.fine_names),
        "coarse_names": list(split.coarse_names),
        "classes": {str(f): {"fine_id": c.fine_id, "coarse_id": c.coarse_id, "split": c.split,
                             "src_support": c.src_support.tolist(), "src_query": c.src_query.tolist(),
                             "tgt_support_reserve": c.tgt_support_reserve.tolist(),
                             "tgt_query": c.tgt_query.tolist()}
                    for f, c in sorted(split.classes.items())},
    }
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(payload, fo, ensure_ascii=False, separators=(",", ":"))


def load_domainshift(path: str) -> DomainShiftSplit:
    with open(path, "r", encoding="utf-8") as fo:
        p = json.load(fo)
    c = p["config"]
    cfg = DomainShiftConfig(
        seed=c["seed"], n_train_super=c["n_train_super"], n_test_super=c["n_test_super"],
        src_pool=c["src_pool"], src_support_size=c["src_support_size"],
        src_query_size=c["src_query_size"], k_shots=tuple(c["k_shots"]),
        corruptions=tuple(c["corruptions"]), severity=c["severity"])
    classes = {int(f): ClassStreams(
        fine_id=d["fine_id"], coarse_id=d["coarse_id"], split=d["split"],
        src_support=np.asarray(d["src_support"], np.int32),
        src_query=np.asarray(d["src_query"], np.int32),
        tgt_support_reserve=np.asarray(d["tgt_support_reserve"], np.int32),
        tgt_query=np.asarray(d["tgt_query"], np.int32))
        for f, d in p["classes"].items()}
    return DomainShiftSplit(cfg, {k: list(v) for k, v in p["superclass_split"].items()},
                            classes, tuple(p["fine_names"]), tuple(p["coarse_names"]))
