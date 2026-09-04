"""Swappability check: change the diffusion backbone and the meta layer needs no edit.

    python -m tests.test_backbone_swap

This file defines, on the spot, two backbones **entirely unlike** the reference U-Net:

  - TinyDiT        : token based (patchify -> transformer -> unpatchify), showing that the
                     contract clause "unpatchify back to a spatial feature map" is workable
  - PretrainedLike : carries its own eps_hat head (provides_eps=True), simulating an already
                     trained DDPM, whose eps_hat ScoreModel reuses while adding a basis head

The **same** meta-layer code then runs each of the three backbones,
    SetEncoder → z_S → Transport → z̃_T → ScoreModel.eps_hat(x_t, t, z̃_T)
verifying that equations (21) and (19) hold, shapes agree, and the adapted freedom equals k.
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch import Tensor, nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusion.forward import q_sample                                      # noqa: E402
from diffusion.schedule import NoiseSchedule                                # noqa: E402
from models.backbone import (                                               # noqa: E402
    BackboneSpec, DiffusionBackbone, build_backbone, check_backbone, register_backbone,
)
from models.score_model import ScoreModel                                   # noqa: E402
from models.set_encoder import SetEncoder                                   # noqa: E402
from models.transport import Transport                                      # noqa: E402
from models.unet import SmallUNet, timestep_embedding                       # noqa: E402

RULE = "─" * 78
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ——  {detail}" if detail else ""))


# ===========================================================================
# Backbone A: token based
# ===========================================================================

@register_backbone("tiny_dit")
class TinyDiT(DiffusionBackbone):
    """patchify -> transformer -> unpatchify. Structurally nothing in common with a U-Net."""

    def __init__(
        self, image_channels: int = 3, image_size: int = 32, patch: int = 4,
        dim: int = 128, depth: int = 2, heads: int = 4, feature_channels: int = 48,
        max_timestep: int = 999,
    ) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        self.patch, self.dim = patch, dim
        self.grid = image_size // patch
        self.feature_channels = feature_channels
        self._spec = BackboneSpec(feature_channels, image_channels, image_size, provides_eps=False)

        self.patch_embed = nn.Conv2d(image_channels, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.to_pixels = nn.Linear(dim, feature_channels * patch * patch)

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def forward_features(self, x_t: Tensor, t: Tensor) -> Tensor:
        b = x_t.shape[0]
        h = self.patch_embed(x_t).flatten(2).transpose(1, 2) + self.pos      # (B, N, D)
        h = h + self.time_mlp(timestep_embedding(t, self.dim)).unsqueeze(1)  # time broadcast over tokens
        for blk in self.blocks:
            h = blk(h)
        h = self.to_pixels(self.norm(h))                                     # (B, N, C*p*p)

        # unpatchify back to a spatial feature map -- the step the contract requires
        p, g, c = self.patch, self.grid, self.feature_channels
        h = h.reshape(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return h.reshape(b, c, g * p, g * p)


# ===========================================================================
# Backbone B: carries its own eps_hat head (simulating a pretrained DDPM)
# ===========================================================================

@register_backbone("pretrained_like")
class PretrainedLike(DiffusionBackbone):
    """A model whose eps_hat is already trained. ScoreModel should reuse it, not add a base head."""

    def __init__(self, image_channels: int = 3, image_size: int = 32,
                 width: int = 32, max_timestep: int = 999) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        self._spec = BackboneSpec(width, image_channels, image_size, provides_eps=True)
        self.time = nn.Linear(width, width)
        self.body = nn.Sequential(
            nn.Conv2d(image_channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
        )
        self.eps_head = nn.Conv2d(width, image_channels, 3, padding=1)   # its own eps_hat head
        self.width = width

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def forward_features(self, x_t: Tensor, t: Tensor) -> Tensor:
        temb = self.time(timestep_embedding(t, self.width))[:, :, None, None]
        return self.body(x_t) + temb

    def forward_eps(self, x_t: Tensor, t: Tensor, feats: Tensor | None = None) -> Tensor:
        return self.eps_head(self.forward_features(x_t, t) if feats is None else feats)


# ===========================================================================

def main() -> int:
    torch.manual_seed(0)
    print(RULE)
    print("backbone swappability check")
    print(RULE)

    B, C, S, K = 3, 3, 32, 16
    sched = NoiseSchedule(n_steps=1000, kind="cosine")

    print("\n[1] contract check (check_backbone)")
    backbones = {
        "small_unet":      build_backbone("small_unet", image_channels=C, image_size=S),
        "tiny_dit":        build_backbone("tiny_dit", image_channels=C, image_size=S),
        "pretrained_like": build_backbone("pretrained_like", image_channels=C, image_size=S),
    }
    for name, bb in backbones.items():
        check_backbone(bb)
    record("all three backbones satisfy the contract", True, " / ".join(backbones))
    record("registry constructs by name", set(backbones) <= set(__import__("models.backbone", fromlist=["BACKBONES"]).BACKBONES))

    # ---- The same meta-layer code across three backbones ----
    print("\n[2] the same meta-layer code across three backbones")
    print(f"  {'backbone':<18}{'C_feat':>8}{'phi params':>12}{'adapted':>12}{'base head':>12}")
    print("  " + "-" * 64)

    x0 = torch.randn(B, C, S, S)
    nb = q_sample(sched, x0, torch.tensor([0, 500, 999]), torch.randn(B, C, S, S))
    support = torch.randn(40, C, S, S)          # one source support set
    relation = torch.tensor([7])                # relation id = superclass id

    outs, ok_21, ok_19, ok_k = {}, True, True, True
    for name, bb in backbones.items():
        # From here to the end of the loop, no line depends on the specific backbone
        model = ScoreModel(bb, sched, k=K).eval()
        encoder = SetEncoder(image_channels=C, image_size=S, k=K).eval()
        transport = Transport(k=K, n_relations=20).eval()

        with torch.no_grad():
            z_s = encoder(support).unsqueeze(0).expand(B, -1)
            z_t = transport(z_s, relation.expand(B))
            eps = model.eps_hat(nb.x_t, nb.t, z_t)
            base, residual = model.split_eps(nb.x_t, nb.t, z_t)
            score = model.score(nb.x_t, nb.t, z_t)
            R = model.basis(nb.x_t, nb.t)
        # End of the backbone-independent section

        ok_21 &= torch.allclose(eps, base + residual, atol=1e-5)
        ok_19 &= torch.allclose(score, -eps / sched.sigma_t(nb.t, 4), atol=1e-6)
        n = model.n_parameters()
        ok_k &= n["adapted_at_deployment"] == K
        outs[name] = (eps, R, n, model)

        print(f"  {name:<18}{bb.spec.feature_channels:>8}{n['total_phi']/1e6:>11.2f}M"
              f"{n['adapted_at_deployment']:>12}"
              f"{('from backbone' if model.base_head is None else 'new'):>12}")

    record("eps_hat shape identical across backbones",
           len({tuple(v[0].shape) for v in outs.values()}) == 1,
           str(tuple(next(iter(outs.values()))[0].shape)))
    record("basis shape identical across backbones (B,k,C,H,W)",
           len({tuple(v[1].shape) for v in outs.values()}) == 1,
           str(tuple(next(iter(outs.values()))[1].shape)))
    record("equation (21) holds for all three backbones", ok_21)
    record("equation (19) holds for all three backbones", ok_19)
    record("adapted freedom is always k, independent of the backbone", ok_k, f"k = {K}")

    # ---- The provides_eps branch ----
    print("\n[3] a backbone carrying its own eps_hat head (the pretrained path)")
    m_pre = outs["pretrained_like"][3]
    record("no base head is created when provides_eps=True", m_pre.base_head is None)
    record("base head parameter count is 0", m_pre.n_parameters()["base_head"] == 0)
    with torch.no_grad():
        direct = m_pre.backbone.forward_eps(nb.x_t, nb.t)
        via_meta = m_pre.eps_base(nb.x_t, nb.t)
    record("eps_base reuses the backbone eps_hat", torch.allclose(direct, via_meta, atol=0, rtol=0),
           "bit-for-bit equal")
    m_unet = outs["small_unet"][3]
    record("a base head is created when provides_eps=False", m_unet.base_head is not None,
           f"{m_unet.n_parameters()['base_head']} parameters")

    # ---- Contract violations must be rejected ----
    print("\n[4] contract violations are rejected")

    class BadShape(DiffusionBackbone):
        max_timestep = 999
        @property
        def spec(self): return BackboneSpec(16, 3, 32)
        def forward_features(self, x_t, t):        # wrong shape: spatial dimensions missing
            return torch.randn(x_t.shape[0], 16)

    class TimeBlind(DiffusionBackbone):
        max_timestep = 999
        def __init__(self):
            super().__init__(); self.c = nn.Conv2d(3, 16, 3, padding=1)
        @property
        def spec(self): return BackboneSpec(16, 3, 32)
        def forward_features(self, x_t, t):        # ignores t
            return self.c(x_t)

    for cls, why in ((BadShape, "feature shape violates the contract"), (TimeBlind, "time embedding not wired")):
        try:
            check_backbone(cls(), verbose=False)
            record(f"rejected: {why}", False, "no error raised")
        except Exception as e:
            record(f"rejected: {why}", True, type(e).__name__)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(RULE)
    print(f"{len(results) - n_fail} / {len(results)} checks passed" + ("" if not n_fail else f"  --  {n_fail} failed"))
    print(RULE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
