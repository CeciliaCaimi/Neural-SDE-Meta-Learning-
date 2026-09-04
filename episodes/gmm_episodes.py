"""Episodes for stage 1.

One difference from the CIFAR version: samples are **drawn on demand** rather than indexed
from a fixed pool. Independent draws from a continuous distribution never coincide, so the
four streams are disjoint automatically and no index assertion is needed.

Field names match EpisodeBatch exactly, so training/meta_train.py and diagnostics/controls.py
work unchanged. One extra field, task, carries the analytic ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from domains.gmm2d import GMMTask


def as_batch(x: Tensor) -> Tensor:
    """(N, 2) -> (N, 2, 1, 1), matching the DiffusionBackbone shape contract."""
    return x.reshape(x.shape[0], -1, 1, 1)


def as_points(x: Tensor) -> Tensor:
    """(N, 2, 1, 1) -> (N, 2)"""
    return x.reshape(x.shape[0], -1)


@dataclass(frozen=True)
class GMMEpisodeBatch:
    src_support: Tensor    # (M_S, 2, 1, 1)
    src_query: Tensor      # (q,   2, 1, 1)
    tgt_support: Tensor    # (K_T, 2, 1, 1)
    tgt_query: Tensor      # (q,   2, 1, 1)
    relation: Tensor       # () long
    task: GMMTask          # analytic ground truth

    @property
    def provenance(self) -> str:
        return f"task{self.task.task_id}/{self.task.relation}/K={self.tgt_support.shape[0]}"


class GMMEpisodeLoader:
    def __init__(
        self,
        tasks: list[GMMTask],
        device: torch.device | str = "cuda",
        m_source: int = 256,
        query_batch: int = 256,
        k_shots: tuple[int, ...] = (1, 2, 5, 10, 20),
        seed: int = 0,
    ) -> None:
        self.tasks = [t.to(device) for t in tasks]
        self.device = torch.device(device)
        self.m_source = int(m_source)
        self.query_batch = int(query_batch)
        self.k_shots = tuple(k_shots)
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, k_shot: int | None = None, task_idx: int | None = None) -> GMMEpisodeBatch:
        i = self.rng.integers(len(self.tasks)) if task_idx is None else task_idx
        task = self.tasks[i]
        k = int(k_shot if k_shot is not None else self.rng.choice(self.k_shots))
        return GMMEpisodeBatch(
            src_support=as_batch(task.source.sample(self.m_source, self.gen)),
            src_query=as_batch(task.source.sample(self.query_batch, self.gen)),
            tgt_support=as_batch(task.target.sample(k, self.gen)),
            tgt_query=as_batch(task.target.sample(self.query_batch, self.gen)),
            relation=torch.tensor(task.relation_id, device=self.device, dtype=torch.long),
            task=task,
        )

    def sample_many(self, n: int, k_shot: int | None = None) -> list[GMMEpisodeBatch]:
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.tasks) * len(self.k_shots)
