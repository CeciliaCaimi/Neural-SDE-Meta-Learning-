"""Episode -> GPU tensors.

All of CIFAR-100 is 60000 x 32 x 32 x 3 uint8 = 184 MB, so it stays resident on the GPU and
each draw is a GPU-side index: the training loop performs no host-to-device transfer.

The four streams are used strictly as the document requires:
  support sets compute z only, query sets compute losses only. EpisodeBatch keeps them
  apart so each downstream function can take only what it is entitled to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from domains.cifar100 import CIFAR100Raw
from episodes.sampler import make_episode
from episodes.splits import MetaSplit
from episodes.types import Provenance


@dataclass(frozen=True)
class EpisodeBatch:
    """Every tensor one training step needs, normalised to [-1, 1]."""

    src_support: Tensor    # (m_enc, C, H, W)  -> r_ψ -> z_S
    src_query: Tensor      # (q, C, H, W)      -> L_src
    tgt_support: Tensor    # (K_T, C, H, W)    -> r_psi -> z^enc_T (baseline branch)
    tgt_query: Tensor      # (q, C, H, W)      -> L_tgt / L_trans
    relation: Tensor       # () long; relation id = superclass id
    provenance: Provenance


class EpisodeLoader:
    """Sample episodes by (source, target) pair and K_T, then draw minibatches from the pools."""

    def __init__(
        self,
        raw: CIFAR100Raw,
        split: MetaSplit,
        which: str,
        device: torch.device | str = "cuda",
        enc_source_images: int = 64,
        query_batch: int = 32,
        k_shots: tuple[int, ...] | None = None,
        seed: int = 0,
        pin_to_device: bool = True,
    ) -> None:
        self.split = split
        self.which = which
        self.device = torch.device(device)
        self.enc_source_images = int(enc_source_images)
        self.query_batch = int(query_batch)
        self.k_shots = tuple(k_shots or split.config.k_shots)
        self.pairs = split.pairs(which)
        if not self.pairs:
            raise ValueError(f"split '{which}' contains no (source, target) combination")

        self.rng = np.random.default_rng(seed)
        # The whole uint8 dataset stays resident on the target device
        imgs = torch.from_numpy(np.ascontiguousarray(raw.images))       # (N,H,W,C) uint8
        self.images = imgs.to(self.device) if pin_to_device else imgs
        self._on_device = pin_to_device

    # ---- Pixel fetch --------------------------------------------------------

    def _fetch(self, idx: np.ndarray) -> Tensor:
        """Global image ids -> (N, C, H, W) float in [-1, 1]."""
        gi = torch.from_numpy(np.ascontiguousarray(idx)).to(self.images.device).long()
        x = self.images.index_select(0, gi)                              # (N,H,W,C) uint8
        x = x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        return x.to(self.device, non_blocking=True) if not self._on_device else x

    def _subsample(self, pool: np.ndarray, n: int) -> np.ndarray:
        if n >= pool.size:
            return pool
        return self.rng.choice(pool, size=n, replace=False)

    # ---- Sampling -----------------------------------------------------------

    def sample(self, k_shot: int | None = None) -> EpisodeBatch:
        coarse_id, s_fine, t_fine = self.pairs[self.rng.integers(len(self.pairs))]
        k = int(k_shot if k_shot is not None else self.rng.choice(self.k_shots))
        ep = make_episode(self.split, coarse_id, s_fine, t_fine, k)

        return EpisodeBatch(
            src_support=self._fetch(self._subsample(ep.src_support, self.enc_source_images)),
            src_query=self._fetch(self._subsample(ep.src_query, self.query_batch)),
            tgt_support=self._fetch(ep.tgt_support),                     # all K_T images
            tgt_query=self._fetch(self._subsample(ep.tgt_query, self.query_batch)),
            relation=torch.tensor(ep.relation, device=self.device, dtype=torch.long),
            provenance=ep.provenance,
        )

    def sample_many(self, n: int, k_shot: int | None = None) -> list[EpisodeBatch]:
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.pairs) * len(self.k_shots)
