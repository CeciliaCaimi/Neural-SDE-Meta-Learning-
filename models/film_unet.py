"""A2's comparison arm backbone: generic latent conditioning by FiLM.

The point of the comparison is to hold the coordinate machinery fixed -- the *same*
SetEncoder emits the *same* k-dimensional z, the *same* Transport moves it -- and change
**only** how z reaches the denoiser. Instead of the additive low-rank basis of equation
(21),

    eps_hat_z = eps_hat_0 + sum_l z_l R_l(x_t, t),                    (21, the paper's model)

z modulates the U-Net's internal features. If FiLM matches or beats the structured basis,
the additive reverse-dynamics hypothesis earned nothing; if it does not, the basis is doing
real work. z enters through forward_features here rather than through a head, which is why
baselines/film_conditioning.py drops the basis term.

Two places z can enter, selected by ``film_mode``:

``per_block`` (default, and what A2 reports)
    Every residual block gets its own affine projection of the coordinate,

        [gamma_j(z), beta_j(z)] = W_j z,   gamma_j, beta_j in R^{C_j},

    added to the timestep scale-shift at the same site, so the block computes

        (1 + scale_t + gamma_j(z)) * GroupNorm(h) + shift_t + beta_j(z),

    with gamma_j and beta_j broadcast over the spatial dimensions. This is FiLM as
    ordinarily specified, and it is the stronger arm: each block reads z through its own
    matrix rather than through one shared embedding.

    The projection carries no bias. A constant b_j would be exactly redundant with the
    bias emb_proj already emits into the same scale and shift, so it buys no capacity,
    and it would cost the z = 0 control its meaning: with b_j free, z = 0 drifts away from
    the unconditioned network during training, while in the additive arm the basis term
    vanishes at z = 0 by construction. Matching that is worth more than a redundant
    parameter. See ResBlock.attach_film.

``temb``
    z is pushed through a two-layer MLP and added to the timestep embedding, so every
    block sees the same function of z through the projection it already uses for t. This
    is what E15 measured on the denoising loss (handoff/Jing-Peng/results/E15_film_panel.txt);
    it is kept so that number stays reproducible, not because A2 reports it.

Both modes are the identity modulation at initialisation -- ``per_block`` zero-initialises
every W_j, ``temb`` zero-initialises the last layer of z_mlp -- so check_backbone, which
passes no z, sees the plain unconditioned U-Net and matches SmallUNet bit-for-bit.

They part once trained, and that is the second reason A2 reports ``per_block``. There z = 0
stays exactly the unconditioned network for the life of the run, as the additive basis
does. Under ``temb`` it does not: z_mlp's first layer keeps a bias, so z_mlp(0) drifts from
zero as soon as the second layer's weight moves, and the z = 0 control quietly becomes "the
backbone plus a learned constant" instead of "the backbone". Both arms have to give z = 0
the same meaning or the no-adaptation row is not a shared baseline.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.backbone import register_backbone
from models.unet import ResBlock, SmallUNet, timestep_embedding

FILM_MODES = ("per_block", "temb")


@register_backbone("film_unet")
class FiLMUNet(SmallUNet):
    """SmallUNet whose features are modulated by z (FiLM).

    Reuses SmallUNet's trunk unchanged; the only structural addition is the z projections.
    ``k`` is the coordinate dimension and must match cfg.model.k (runner/train_film.py
    injects it). forward_features keeps z optional so the DiffusionBackbone contract check
    -- which calls forward_features(x, t) with no z -- still exercises the network at z = 0.
    """

    #: forward_features takes a third argument. FiLMScoreModel refuses to build on a
    #: backbone without this, rather than silently conditioning on nothing.
    accepts_z = True

    def __init__(self, k: int = 16, film_mode: str = "per_block", **unet_kwargs) -> None:
        super().__init__(**unet_kwargs)
        if k <= 0:
            raise ValueError("k must be positive")
        if film_mode not in FILM_MODES:
            raise ValueError(f"film_mode must be one of {FILM_MODES}, got {film_mode!r}")
        self.k = int(k)
        self.film_mode = film_mode

        if film_mode == "per_block":
            # One projection per residual block, at the sites already used for timestep
            # conditioning. Attached after the trunk is built so the block list is final.
            for m in self.modules():
                if isinstance(m, ResBlock):
                    m.attach_film(self.k)
            self.z_mlp = None
        else:
            # z -> temb offset. Same width as the timestep embedding so it adds directly.
            self.z_mlp = nn.Sequential(
                nn.Linear(self.k, self.temb_dim), nn.SiLU(),
                nn.Linear(self.temb_dim, self.temb_dim),
            )
            nn.init.zeros_(self.z_mlp[-1].weight)
            nn.init.zeros_(self.z_mlp[-1].bias)

    def film_parameters(self) -> list[nn.Parameter]:
        """Exactly the parameters by which z reaches the denoiser. Reported beside the
        basis head's parameter count, which is the matched quantity."""
        if self.film_mode == "per_block":
            return [p for m in self.modules()
                    if isinstance(m, ResBlock) and m.z_proj is not None
                    for p in m.z_proj.parameters()]
        return list(self.z_mlp.parameters())

    def forward_features(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        if z is not None and z.dim() == 1:
            z = z.unsqueeze(0).expand(x_t.shape[0], -1)
        if self.film_mode == "temb":
            if z is not None:
                temb = temb + self.z_mlp(z.to(temb.dtype))
            return self._trunk(x_t, temb)
        return self._trunk(x_t, temb, z)
