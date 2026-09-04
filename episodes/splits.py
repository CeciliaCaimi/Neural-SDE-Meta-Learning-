"""The semantic-task split protocol.

Rules (set by the project; these differ from the clean->corrupted scheme of section 12.3):

  1. The 20 superclasses are split into meta-train / val / test, mutually disjoint.
  2. The 5 fine classes inside each superclass are then given roles: some are source
     (keeping their full image pool), the rest target (support capped artificially at K_T).
     CIFAR-100 is perfectly balanced at 600 images per fine class, so "many/few photos" is
     **constructed** here, matching the "artificially restricted target support set" of 6.1.
  3. source support / source query / target support / target query are pairwise disjoint by
     **global image id**. The target support is **nested** over K_T in {1,2,5,10,20}: the
     image used at K_T=1 also lies in the K_T=2 set, so the K_T sweep is a within-group
     comparison on one query batch and is never confounded by which images were drawn.

All randomness comes from RNGs keyed by (seed, object id), so changing one configuration
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from domains.cifar100 import (
    FINE_PER_COARSE,
    IMAGES_PER_FINE,
    N_COARSE,
    CIFAR100Raw,
)

SPLIT_NAMES = ("train", "val", "test")
ROLE_SOURCE = "source"
ROLE_TARGET = "target"


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 12345

    # ---- Superclass level ----
    # The validation split is required by the protocol: every hyperparameter and adaptation
    # budget is chosen on val classes and then frozen; test serves final reporting only.
    n_train_super: int = 12          # set by the project
    n_val_super: int = 4
    n_test_super: int = 4

    # ---- Role assignment inside a superclass ----
    n_source_per_super: int = 3      # the remaining 5 - 3 = 2 fine classes take the target role

    # ---- Image pool of a source-role fine class ----
    src_support_size: int = 500
    src_query_size: int = 100

    # ---- Image pool of a target-role fine class ----
    k_shots: tuple[int, ...] = (1, 2, 5, 10, 20)   # section 12.3
    # target support reserve = max(k_shots); the query takes all that remains

    def __post_init__(self) -> None:
        total = self.n_train_super + self.n_val_super + self.n_test_super
        if total != N_COARSE:
            raise ValueError(f"superclass allocation should sum to {N_COARSE}, got {total}")
        if not 1 <= self.n_source_per_super < FINE_PER_COARSE:
            raise ValueError(f"n_source_per_super must lie in 1..{FINE_PER_COARSE - 1}")
        if self.src_support_size + self.src_query_size > IMAGES_PER_FINE:
            raise ValueError("source pool exceeds the 600 images available per class")
        if self.tgt_support_reserve + 1 > IMAGES_PER_FINE:
            raise ValueError("target support reserve exceeds the 600 images available per class")

    @property
    def tgt_support_reserve(self) -> int:
        return max(self.k_shots)

    @property
    def n_target_per_super(self) -> int:
        return FINE_PER_COARSE - self.n_source_per_super


@dataclass(frozen=True)
class ClassPools:
    """Pool allocation for one fine class. All indices are global image ids from CIFAR100Raw."""

    fine_id: int
    coarse_id: int
    role: str                       # ROLE_SOURCE | ROLE_TARGET
    split: str                      # train | val | test
    src_support: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    src_query: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    tgt_support_reserve: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    tgt_query: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))

    def support_for_k(self, k: int) -> np.ndarray:
        """K_T target support images: the first k of the reserve, hence nested in k."""
        if self.role != ROLE_TARGET:
            raise ValueError(f"fine class {self.fine_id} has role {self.role} and no target support")
        if k > self.tgt_support_reserve.size:
            raise ValueError(f"K_T={k} exceeds the reserve of {self.tgt_support_reserve.size}")
        return self.tgt_support_reserve[:k]


@dataclass(frozen=True)
class MetaSplit:
    config: SplitConfig
    superclass_split: dict[str, list[int]]        # split -> [coarse_id]
    pools: dict[int, ClassPools]                  # fine_id -> pools
    fine_names: tuple[str, ...]
    coarse_names: tuple[str, ...]

    def split_of_coarse(self, coarse_id: int) -> str:
        for name, ids in self.superclass_split.items():
            if coarse_id in ids:
                return name
        raise KeyError(coarse_id)

    def fine_ids(self, split: str, role: str | None = None) -> list[int]:
        out = [
            f for f, p in self.pools.items()
            if p.split == split and (role is None or p.role == role)
        ]
        return sorted(out)

    def pairs(self, split: str) -> list[tuple[int, int, int]]:
        """All (coarse_id, source_fine, target_fine) combinations within this split.

        Source and target must share a superclass and must be different fine classes.
        """
        out = []
        for coarse_id in sorted(self.superclass_split[split]):
            srcs = [f for f in self.fine_ids(split, ROLE_SOURCE)
                    if self.pools[f].coarse_id == coarse_id]
            tgts = [f for f in self.fine_ids(split, ROLE_TARGET)
                    if self.pools[f].coarse_id == coarse_id]
            for s in srcs:
                for t in tgts:
                    out.append((coarse_id, s, t))
        return out


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _rng(seed: int, *stream: int) -> np.random.Generator:
    """Independent random stream keyed by (seed, object id); one config change disturbs no other pool."""
    return np.random.default_rng([seed, *stream])


def build_split(raw: CIFAR100Raw, cfg: SplitConfig | None = None) -> MetaSplit:
    cfg = cfg or SplitConfig()

    # ---- 1. Three-way superclass split ----
    order = _rng(cfg.seed, 0).permutation(N_COARSE)
    n_tr, n_va = cfg.n_train_super, cfg.n_val_super
    superclass_split = {
        "train": sorted(int(c) for c in order[:n_tr]),
        "val":   sorted(int(c) for c in order[n_tr:n_tr + n_va]),
        "test":  sorted(int(c) for c in order[n_tr + n_va:]),
    }

    # ---- 2+3. Assign roles inside each superclass, then split each fine class pool ----
    pools: dict[int, ClassPools] = {}
    for split_name, coarse_ids in superclass_split.items():
        for coarse_id in coarse_ids:
            fines = np.flatnonzero(raw.fine_to_coarse == coarse_id)
            # Roles: shuffle with the (seed, 1000+coarse_id) stream; the first n_source are source
            shuffled = _rng(cfg.seed, 1000 + coarse_id).permutation(fines)
            src_fines = sorted(int(f) for f in shuffled[:cfg.n_source_per_super])
            tgt_fines = sorted(int(f) for f in shuffled[cfg.n_source_per_super:])

            for fine_id in src_fines:
                idx = _shuffled_pool(raw, cfg, fine_id)
                a, b = cfg.src_support_size, cfg.src_support_size + cfg.src_query_size
                pools[fine_id] = ClassPools(
                    fine_id=fine_id, coarse_id=coarse_id,
                    role=ROLE_SOURCE, split=split_name,
                    src_support=idx[:a], src_query=idx[a:b],
                )

            for fine_id in tgt_fines:
                idx = _shuffled_pool(raw, cfg, fine_id)
                r = cfg.tgt_support_reserve
                pools[fine_id] = ClassPools(
                    fine_id=fine_id, coarse_id=coarse_id,
                    role=ROLE_TARGET, split=split_name,
                    tgt_support_reserve=idx[:r], tgt_query=idx[r:],
                )

    return MetaSplit(
        config=cfg,
        superclass_split=superclass_split,
        pools=pools,
        fine_names=raw.fine_names,
        coarse_names=raw.coarse_names,
    )


def _shuffled_pool(raw: CIFAR100Raw, cfg: SplitConfig, fine_id: int) -> np.ndarray:
    """Deterministic shuffle of one fine class's 600 images, seeded only by (seed, fine_id)."""
    idx = raw.indices_of_fine(fine_id)
    assert idx.size == IMAGES_PER_FINE, (fine_id, idx.size)
    return _rng(cfg.seed, 2000 + fine_id).permutation(idx).astype(np.int32)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_split(split: MetaSplit, path: str) -> None:
    payload = {
        "config": asdict(split.config),
        "superclass_split": split.superclass_split,
        "fine_names": list(split.fine_names),
        "coarse_names": list(split.coarse_names),
        "pools": {
            str(f): {
                "fine_id": p.fine_id, "coarse_id": p.coarse_id,
                "role": p.role, "split": p.split,
                "src_support": p.src_support.tolist(),
                "src_query": p.src_query.tolist(),
                "tgt_support_reserve": p.tgt_support_reserve.tolist(),
                "tgt_query": p.tgt_query.tolist(),
            }
            for f, p in sorted(split.pools.items())
        },
    }
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(payload, fo, ensure_ascii=False, separators=(",", ":"))


def load_split(path: str) -> MetaSplit:
    with open(path, "r", encoding="utf-8") as fo:
        payload = json.load(fo)
    c = payload["config"]
    cfg = SplitConfig(
        seed=c["seed"],
        n_train_super=c["n_train_super"], n_val_super=c["n_val_super"],
        n_test_super=c["n_test_super"],
        n_source_per_super=c["n_source_per_super"],
        src_support_size=c["src_support_size"], src_query_size=c["src_query_size"],
        k_shots=tuple(c["k_shots"]),
    )
    pools = {
        int(f): ClassPools(
            fine_id=d["fine_id"], coarse_id=d["coarse_id"],
            role=d["role"], split=d["split"],
            src_support=np.asarray(d["src_support"], np.int32),
            src_query=np.asarray(d["src_query"], np.int32),
            tgt_support_reserve=np.asarray(d["tgt_support_reserve"], np.int32),
            tgt_query=np.asarray(d["tgt_query"], np.int32),
        )
        for f, d in payload["pools"].items()
    }
    return MetaSplit(
        config=cfg,
        superclass_split={k: list(v) for k, v in payload["superclass_split"].items()},
        pools=pools,
        fine_names=tuple(payload["fine_names"]),
        coarse_names=tuple(payload["coarse_names"]),
    )
