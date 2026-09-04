"""The reusable source-to-target transport Delta_gamma (section 8.3).

    z_tilde_{y,T} = z_{y,S} + Delta_gamma(z_{y,S}, c_{S->T})                  (25)

Three points govern this module:

  1. **The input is only z_S and the relation descriptor c -- never a target
     image.** The transport map must predict the target coordinate having seen
     no target sample at all; that is precisely the proposition under test.
     Target images appear only in (a) the parallel branch that produces
     z^enc_T for the target-only baseline, and (b) meta-test refinement.

  2. **The relation embedding is indexed by relation, never by semantic class.**
     Indexing by class would let Delta_gamma degenerate into memorisation and
     would destroy the "relation generalisation" property of section 15. In the
     domain-shift scheme the relation id is the transformation type, which
     recurs across the train, validation and test classes alike.

  3. The specification comes from A.2: for k = 16, a residual MLP
     (16 + d_c) -> 64 -> 64 -> 16.

Independent of the backbone: it operates purely in the k-dimensional coordinate
space.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Transport(nn.Module):
    def __init__(
        self,
        k: int = 16,
        n_relations: int | None = None,
        relation_dim: int = 8,
        hidden: int = 64,
        out_moments: bool = False,
        init_scale: float = 1e-2,
    ) -> None:
        """
        n_relations : number of relations. None means a single relation is under
                      study, in which case c is omitted (permitted by 8.3).
        out_moments : the **single** extension point reserved by A.7. Should the
                      uncertainty of the source prior later prove to matter, this
                      makes Delta_gamma emit the diagonal variance of
                      (mu_tilde_T, log sigma_tilde^2_T). Keep False in v1.
        """
        super().__init__()
        self.k = int(k)
        self.out_moments = bool(out_moments)
        self.n_relations = n_relations

        self.relation_emb = (
            nn.Embedding(n_relations, relation_dim) if n_relations is not None else None
        )
        in_dim = self.k + (relation_dim if self.relation_emb is not None else 0)
        out_dim = self.k * (2 if out_moments else 1)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        # Small initialisation, so that z_tilde_T is approximately z_S at the
        # start -- equivalent to the "reuse the source coordinate" baseline --
        # and training then pushes it away. This keeps any gain attributable to
        # the transport map itself.
        nn.init.normal_(self.net[-1].weight, std=init_scale)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _as_batched(z_s: Tensor) -> tuple[Tensor, bool]:
        """Normalise to (B, k), recording whether the result must be squeezed
        back to (k,) on return."""
        return (z_s.unsqueeze(0), True) if z_s.dim() == 1 else (z_s, False)

    def delta(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        """Return only the increment Delta, without adding z_S. Used by the
        ablation that reuses z_S directly."""
        z_s, squeeze = self._as_batched(z_s)
        if z_s.shape[-1] != self.k:
            raise ValueError(f"z_S should have dimension {self.k}, got {z_s.shape[-1]}")

        if self.relation_emb is not None:
            if relation is None:
                raise ValueError("this Transport has a relation embedding; a relation id is required")
            if relation.dim() == 0:
                relation = relation.expand(z_s.shape[0])
            h = torch.cat([z_s, self.relation_emb(relation)], dim=-1)
        else:
            if relation is not None:
                raise ValueError("this Transport has no relation embedding; do not pass a relation")
            h = z_s
        out = self.net(h)
        return out.squeeze(0) if squeeze else out

    def forward(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        """Equation (25). A (k,) input returns (k,) and a (B, k) input returns
        (B, k); the shape is never silently changed.

        With out_moments=True the pair (z_tilde_T, log sigma_tilde^2_T) is
        returned instead.
        """
        out = self.delta(z_s, relation)
        if self.out_moments:
            d, logvar = out.chunk(2, dim=-1)
            return z_s + d, logvar
        return z_s + out
