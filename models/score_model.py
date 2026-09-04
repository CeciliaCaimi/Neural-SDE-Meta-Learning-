"""Core interface of the meta layer.

    ε̂_{φ,z}(x_t,t) = ε̂_{φ,0}(x_t,t) + Σ_{ℓ=1..k} z_ℓ · R_{φ,ℓ}(x_t,t)        (21)
    s_{φ,z}(x,t)   = −ε̂_{φ,z}(x,t) / σ_t                                     (19)

**Every** downstream component -- samplers, the training loop, each adaptation strategy,
each diagnostic -- reaches the model only here, so swapping the backbone changes nothing below.

Three benefits follow directly from this abstraction:
  - the z=0 control is one line (pass a zero tensor); shuffled-z likewise
  - basis_usage (section 15) needs eps_hat_0 and the residual separately; both are exposed
  - one features() pass feeds both heads: "one U-Net, not k separate denoisers"
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from diffusion.schedule import NoiseSchedule
from models.backbone import DiffusionBackbone


class ScoreModel(nn.Module):
    def __init__(
        self,
        backbone: DiffusionBackbone,
        schedule: NoiseSchedule,
        k: int = 16,
        basis_init_scale: float = 1e-3,
        coord_decoder: str = "linear",
        decoder_hidden: int = 64,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.backbone = backbone
        self.schedule = schedule
        self.k = int(k)

        # Coordinate decoder h_eta. The matched comparison required by section 3.1:
        #     linear    : Delta_s = B(x,t) z         -- default; the claim of the paper
        #     nonlinear : Delta_s = B(x,t) h_eta(z)  -- separates "the low-dimensional
        #                 hypothesis is wrong" from "a global linear basis is too weak".
        # h_eta(z) = z + z * MLP(z) makes h_eta(0) = 0 exact, so the z=0 control keeps its
        # meaning; linear is the case MLP = 0. h_eta is frozen at deployment; only z is refined.
        if coord_decoder not in ("linear", "nonlinear"):
            raise ValueError(f"coord_decoder must be linear or nonlinear, got {coord_decoder}")
        self.coord_decoder_kind = coord_decoder
        if coord_decoder == "nonlinear":
            self.coord_decoder = nn.Sequential(
                nn.Linear(self.k, decoder_hidden), nn.SiLU(),
                nn.Linear(decoder_hidden, decoder_hidden), nn.SiLU(),
                nn.Linear(decoder_hidden, self.k),
            )
            nn.init.zeros_(self.coord_decoder[-1].weight)   # exactly linear at initialisation
            nn.init.zeros_(self.coord_decoder[-1].bias)
        else:
            self.coord_decoder = None

        spec = backbone.spec
        self.image_channels = spec.image_channels
        self.image_size = spec.image_size

        # On a 1x1 "spatial" extent (vector data such as the R^2 points of stage 1) a 3x3
        # convolution would waste 8/9 of its weights, so drop to 1x1. Images are unaffected.
        ks = 3 if spec.image_size > 1 else 1
        pad = ks // 2

        # base head -- not created when the backbone supplies eps_hat (pretrained models)
        self.base_head = (
            None if spec.provides_eps
            else nn.Conv2d(spec.feature_channels, spec.image_channels, ks, padding=pad)
        )

        # basis head -- emits k x C_img channels at once, reshaped into k residual directions
        self.basis_head = nn.Conv2d(
            spec.feature_channels, self.k * spec.image_channels, ks, padding=pad
        )
        # Small initialisation: z has little influence at the start but the gradient is
        # nonzero -- zero-init would pin dL/dz at 0 and starve the encoder of basis gradient.
        nn.init.normal_(self.basis_head.weight, std=basis_init_scale)
        nn.init.zeros_(self.basis_head.bias)

    # ---- Backbone and the two heads ---------------------------------------

    def features(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.backbone.forward_features(x_t, t)

    def eps_base(self, x_t: Tensor, t: Tensor, feats: Tensor | None = None) -> Tensor:
        """eps_hat_0 -- the baseline noise prediction shared by all tasks."""
        if self.base_head is None:
            eps = self.backbone.forward_eps(x_t, t, feats)
            if eps is None:
                raise RuntimeError("backbone declares provides_eps=True but forward_eps returned None")
            return eps
        return self.base_head(self.features(x_t, t) if feats is None else feats)

    def basis(self, x_t: Tensor, t: Tensor, feats: Tensor | None = None) -> Tensor:
        """R_{1..k} -- (B, k, C_img, H, W). The learned score-perturbation directions."""
        f = self.features(x_t, t) if feats is None else feats
        b = f.shape[0]
        return self.basis_head(f).reshape(b, self.k, self.image_channels, *f.shape[-2:])

    # ---- Composition ------------------------------------------------------

    def _prepare_z(self, z: Tensor | None, batch: int, device, dtype) -> Tensor:
        """Normalise z to (B, k) and pass it through h_eta. None means z=0 (the control)."""
        if z is None:
            zz = torch.zeros(batch, self.k, device=device, dtype=dtype)
        else:
            if z.dim() == 1:
                z = z.unsqueeze(0).expand(batch, -1)
            if z.shape != (batch, self.k):
                raise ValueError(f"z should have shape ({batch}, {self.k}), got {tuple(z.shape)}")
            zz = z.to(device=device, dtype=dtype)
        if self.coord_decoder is None:
            return zz
        return zz + zz * self.coord_decoder(zz)      # h_η(0) = 0

    def eps_hat(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        """Equation (21). One backbone pass, shared by both heads."""
        feats = self.features(x_t, t)
        base = self.eps_base(x_t, t, feats)
        zz = self._prepare_z(z, x_t.shape[0], x_t.device, base.dtype)
        return base + torch.einsum("bk,bkchw->bchw", zz, self.basis(x_t, t, feats))

    def eps_hat_many(self, x_t: Tensor, t: Tensor, zs: Sequence[Tensor | None]) -> list[Tensor]:
        """Several z on the same (x_t, t) batch, with **one** backbone pass.

        L_tgt and L_trans act on the same target-query batch (equations 28 and 29) and
        differ only in z, so this saves one backbone pass and one basis-head pass.
        The z=0 and shuffled-z controls use the same path at almost no cost.
        """
        feats = self.features(x_t, t)
        base = self.eps_base(x_t, t, feats)
        R = self.basis(x_t, t, feats)
        b = x_t.shape[0]
        out = []
        for z in zs:
            zz = self._prepare_z(z, b, x_t.device, base.dtype)
            out.append(base + torch.einsum("bk,bkchw->bchw", zz, R))
        return out

    def score(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        """Equation (19). The model works in eps_hat; this is the score reading of it."""
        return self.schedule.eps_to_score(self.eps_hat(x_t, t, z), t)

    def split_eps(
        self, x_t: Tensor, t: Tensor, z: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return (eps_hat_0, sum_l z_l R_l) separately, for the r_basis diagnostic."""
        feats = self.features(x_t, t)
        base = self.eps_base(x_t, t, feats)
        zz = self._prepare_z(z, x_t.shape[0], x_t.device, base.dtype)
        residual = torch.einsum("bk,bkchw->bchw", zz, self.basis(x_t, t, feats))
        return base, residual

    @torch.no_grad()
    def basis_usage(self, x_t: Tensor, t: Tensor, z: Tensor, eps: float = 1e-8) -> Tensor:
        """r_basis = ||eps_res|| / (||eps_base|| + eps), per sample.

        The first guardrail of section 15: if this ratio approaches 0, or z=0 performs as well
        as the correct z, the low-dimensional structure is unproven -- diagnose, do not scale up.
        """
        base, residual = self.split_eps(x_t, t, z)
        return residual.flatten(1).norm(dim=1) / (base.flatten(1).norm(dim=1) + eps)

    # ---- Parameter groups -------------------------------------------------

    def meta_parameters(self):
        """phi -- backbone plus both heads. Entirely frozen at deployment."""
        return self.parameters()

    def n_parameters(self) -> dict[str, int]:
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_basis = sum(p.numel() for p in self.basis_head.parameters())
        n_base = 0 if self.base_head is None else sum(p.numel() for p in self.base_head.parameters())
        return {
            "backbone": n_backbone, "base_head": n_base, "basis_head": n_basis,
            "total_phi": n_backbone + n_base + n_basis,
            "adapted_at_deployment": self.k,      # this many scalars move at deployment
        }

    def extra_repr(self) -> str:
        return f"k={self.k}, backbone={type(self.backbone).__name__}"
