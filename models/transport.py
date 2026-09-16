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


# ---------------------------------------------------------------------------
# A5: the structure ablation. Is the nonlinear residual map earning its place?
# ---------------------------------------------------------------------------
#
# Three simpler maps from z_S to the predicted target coordinate, all sharing this
# module's interface: forward(z_s, relation) -> (k,) or (B, k), a .delta() that returns
# the increment alone, and a .relation_emb attribute that callers read to decide whether
# a relation id must be passed.
#
# Each is trained the way Transport was -- jointly, through the meta objective -- rather
# than regressed onto z^enc_T afterwards. The transport is supervised by the denoising
# loss on the target branch and by nothing else (training/meta_train.py), and the
# agreement |z_tilde - z_enc| is explicitly not a training signal. Fitting these variants
# by least squares onto z^enc_T would give them a different objective from the map they
# are being compared against, and the comparison would answer the wrong question.
#
# All three start at "reuse z_S", exactly as Transport does with its small final-layer
# initialisation, so no variant begins with an advantage in how far it has already moved.


class IdentityTransport(nn.Module):
    """z_T = z_S. The zero-parameter reference.

    It needs a training run of its own even though it has nothing to learn: the backbone
    and the encoder are shaped by whatever transport sits beside them, so evaluating
    direct reuse on a backbone trained around a map that moves the coordinate would
    measure the wrong thing.
    """

    def __init__(self, k: int = 16, n_relations: int | None = None, **_) -> None:
        super().__init__()
        self.k = int(k)
        self.n_relations = n_relations
        self.out_moments = False
        self.relation_emb = None          # the relation is ignored, by construction
        # A parameter-free module still needs one parameter for the optimiser to accept it
        self.register_buffer("_unused", torch.zeros(1), persistent=False)

    def delta(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        return torch.zeros_like(z_s)

    def forward(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        return z_s


class ConstantRelationTransport(nn.Module):
    """z_T = z_S + delta_c: one learned vector per relation, independent of z_S.

    This is Transport with the dependence on z_S removed. If it matches the nonlinear map
    then the transport is a lookup table over relations and the source coordinate enters
    only as the thing being offset.
    """

    def __init__(self, k: int = 16, n_relations: int | None = None, **_) -> None:
        super().__init__()
        self.k = int(k)
        self.n_relations = n_relations
        self.out_moments = False
        n = 1 if n_relations is None else int(n_relations)
        self.relation_emb = nn.Embedding(n, self.k) if n_relations is not None else None
        self.delta_table = nn.Embedding(n, self.k)
        nn.init.zeros_(self.delta_table.weight)      # start at reuse, as Transport does

    def _index(self, z_s: Tensor, relation: Tensor | None) -> Tensor:
        if self.n_relations is None:
            return torch.zeros(z_s.shape[0], device=z_s.device, dtype=torch.long)
        if relation is None:
            raise ValueError("this transport is indexed by relation; a relation id is required")
        if relation.dim() == 0:
            relation = relation.expand(z_s.shape[0])
        return relation

    def delta(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        z, squeeze = (z_s.unsqueeze(0), True) if z_s.dim() == 1 else (z_s, False)
        if z.shape[-1] != self.k:
            raise ValueError(f"z_S should have dimension {self.k}, got {z.shape[-1]}")
        out = self.delta_table(self._index(z, relation))
        return out.squeeze(0) if squeeze else out

    def forward(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        return z_s + self.delta(z_s, relation)


class LinearTransport(nn.Module):
    """z_T = A z_S + b_c: one shared matrix, one bias per relation.

    A is shared across relations and the bias is not, which is the natural reading of
    "a linear map" for a family of relations; the alternative, a separate A per relation,
    would give this arm more parameters than the nonlinear map it is compared against.
    A starts at the identity and b at zero, so this too begins at reuse.
    """

    def __init__(self, k: int = 16, n_relations: int | None = None, **_) -> None:
        super().__init__()
        self.k = int(k)
        self.n_relations = n_relations
        self.out_moments = False
        n = 1 if n_relations is None else int(n_relations)
        self.relation_emb = nn.Embedding(n, self.k) if n_relations is not None else None
        self.A = nn.Linear(self.k, self.k, bias=False)
        self.bias_table = nn.Embedding(n, self.k)
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.k))
        nn.init.zeros_(self.bias_table.weight)

    def _index(self, z_s: Tensor, relation: Tensor | None) -> Tensor:
        if self.n_relations is None:
            return torch.zeros(z_s.shape[0], device=z_s.device, dtype=torch.long)
        if relation is None:
            raise ValueError("this transport is indexed by relation; a relation id is required")
        if relation.dim() == 0:
            relation = relation.expand(z_s.shape[0])
        return relation

    def delta(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        """The increment alone, so that this arm reports |dz| on the same axis as the others."""
        return self.forward(z_s, relation) - z_s

    def forward(self, z_s: Tensor, relation: Tensor | None = None) -> Tensor:
        z, squeeze = (z_s.unsqueeze(0), True) if z_s.dim() == 1 else (z_s, False)
        if z.shape[-1] != self.k:
            raise ValueError(f"z_S should have dimension {self.k}, got {z.shape[-1]}")
        out = self.A(z) + self.bias_table(self._index(z, relation))
        return out.squeeze(0) if squeeze else out


TRANSPORTS = {
    "residual_mlp": Transport,                  # the current method, equation (25)
    "identity": IdentityTransport,              # z_T = z_S
    "constant": ConstantRelationTransport,      # z_T = z_S + delta_c
    "linear": LinearTransport,                  # z_T = A z_S + b_c
}


def build_transport(kind: str, **kwargs) -> nn.Module:
    """Named lookup, so the variant travels inside the checkpoint's config.

    The kind belongs in the config rather than in a constructor argument at evaluation
    time: every evaluation script rebuilds its modules with build(cfg, device) and then
    loads the checkpoint's weights into them, so a variant that is not recorded in the
    config is either a shape error or, when the shapes happen to agree, a silent one.
    """
    if kind not in TRANSPORTS:
        raise ValueError(f"unknown transport '{kind}'; expected one of {sorted(TRANSPORTS)}")
    return TRANSPORTS[kind](**kwargs)
