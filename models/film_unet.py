"""E15 comparison arm backbone: generic latent conditioning by FiLM.

The point of the comparison (section 3.1 / the "is a global linear basis too weak?"
question) is to hold the coordinate machinery fixed -- the *same* SetEncoder emits the
*same* k-dimensional z, the *same* Transport moves it -- and change **only** how z reaches
the denoiser. Instead of the additive low-rank basis of equation (21),

    eps_hat_z = eps_hat_0 + sum_l z_l R_l(x_t, t),                    (21, the paper's model)

this backbone folds z into the timestep embedding, so z modulates every residual block's
scale-shift (FiLM):

    temb = time_mlp(timestep_embedding(t)) + z_mlp(z)

That is the standard "just condition the network on the task vector" baseline. If it matches
or beats the structured basis, the low-dimensional *additive* hypothesis earned nothing; if
it does not, the basis is doing real work. z enters through forward_features here rather than
through a head, which is exactly why baselines/film_conditioning.py drops the basis term.

z_mlp's last layer is zero-initialised, so at z = 0 (and at initialisation) FiLM is the
identity modulation: check_backbone (which passes no z) and the z = 0 control both see the
plain unconditioned U-Net, matching SmallUNet bit-for-bit.
"""

from __future__ import annotations

from torch import Tensor, nn

from models.backbone import register_backbone
from models.unet import SmallUNet, timestep_embedding


@register_backbone("film_unet")
class FiLMUNet(SmallUNet):
    """SmallUNet whose timestep embedding is additively modulated by z (FiLM).

    Reuses SmallUNet's trunk unchanged; the only structural addition is z_mlp. ``k`` is the
    coordinate dimension and must match cfg.model.k (runner/train_film.py injects it).
    forward_features keeps z optional so the DiffusionBackbone contract check -- which calls
    forward_features(x, t) with no z -- still exercises the network at z = 0.
    """

    def __init__(self, k: int = 16, **unet_kwargs) -> None:
        super().__init__(**unet_kwargs)
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        # z -> temb offset. Same width as the timestep embedding so it adds directly.
        self.z_mlp = nn.Sequential(
            nn.Linear(self.k, self.temb_dim), nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )
        # Zero-init the output so FiLM is the identity at z = 0 and at initialisation:
        # the z = 0 control and the contract check both reduce to the plain U-Net.
        nn.init.zeros_(self.z_mlp[-1].weight)
        nn.init.zeros_(self.z_mlp[-1].bias)

    def forward_features(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        if z is not None:
            if z.dim() == 1:
                z = z.unsqueeze(0).expand(x_t.shape[0], -1)
            temb = temb + self.z_mlp(z.to(temb.dtype))
        return self._trunk(x_t, temb)
