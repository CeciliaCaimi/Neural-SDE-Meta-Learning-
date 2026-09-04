"""Reference backbone: a small 32x32 DDPM U-Net.

Section A.2 is explicit: "Architecture settings are validation choices, not contributions."
So this aims only to be adequate, runnable and replaceable: it is one **example**
implementation of DiffusionBackbone, not part of the framework. To use a DiT, a larger
U-Net or a pretrained model, write one against the backbone.py contract and register it.

Note: this module contains **no** eps_hat head. Both heads belong to the meta layer
(models/score_model.py), which is what makes "one U-Net, not k denoisers" possible.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.backbone import BackboneSpec, DiffusionBackbone, register_backbone


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch, eps=1e-6)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv twice, with the timestep injected as a scale-shift."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(_norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb_proj = nn.Linear(temb_dim, 2 * out_ch)
        self.out_norm = _norm(out_ch)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Small but nonzero initialisation of the residual branch: it controls the variance
        # accumulated over 16 blocks without making the whole network the identity at
        # initialisation, which would stop the time embedding reaching the features.
        nn.init.normal_(self.out_layers[-1].weight, std=0.02)
        nn.init.zeros_(self.out_layers[-1].bias)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_proj(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class AttnBlock(nn.Module):
    """Single-head self-attention, used only at low and middle resolutions."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)   # (b, hw, hw)
        out = (v @ att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class _Stage(nn.Module):
    """A sequence of modules; ResBlocks receive temb, the others do not."""

    def __init__(self, *mods: nn.Module) -> None:
        super().__init__()
        self.mods = nn.ModuleList(mods)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        for m in self.mods:
            x = m(x, temb) if isinstance(m, ResBlock) else m(x)
        return x


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


@register_backbone("small_unet")
class SmallUNet(DiffusionBackbone):
    """A compact U-Net for 32x32. Outputs **feature maps**, not eps_hat."""

    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 32,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        max_timestep: int = 999,
    ) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        self._spec = BackboneSpec(
            feature_channels=base_channels,
            image_channels=image_channels,
            image_size=image_size,
            provides_eps=False,          # the base head is the meta layer's job
        )

        temb_dim = base_channels * 4
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        # ---- Downsampling path ----
        self.downs = nn.ModuleList()
        skip_chans = [base_channels]
        ch, res = base_channels, image_size
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                mods: list[nn.Module] = [ResBlock(ch, out_ch, temb_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    mods.append(AttnBlock(ch))
                self.downs.append(_Stage(*mods))
                skip_chans.append(ch)
            if i != len(channel_mult) - 1:
                self.downs.append(_Stage(Downsample(ch)))
                skip_chans.append(ch)
                res //= 2

        # ---- Middle ----
        self.mid = _Stage(
            ResBlock(ch, ch, temb_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, temb_dim, dropout)
        )

        # ---- Upsampling path ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                mods = [ResBlock(ch + skip_chans.pop(), out_ch, temb_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    mods.append(AttnBlock(ch))
                if i and j == num_res_blocks:
                    mods.append(Upsample(ch))
                    res *= 2
                self.ups.append(_Stage(*mods))

        assert not skip_chans, f"unconsumed skip connections: {skip_chans}"
        self.out_norm = _norm(ch)
        assert ch == base_channels, (ch, base_channels)

    @property
    def spec(self) -> BackboneSpec:
        return self._spec

    def forward_features(self, x_t: Tensor, t: Tensor) -> Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.conv_in(x_t)
        skips = [h]
        for stage in self.downs:
            h = stage(h, temb)
            skips.append(h)
        h = self.mid(h, temb)
        for stage in self.ups:
            h = stage(torch.cat([h, skips.pop()], dim=1), temb)
        return F.silu(self.out_norm(h))
