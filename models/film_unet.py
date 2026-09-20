"""E15 comparison arm backbone: generic latent conditioning by FiLM.

The comparison holds the coordinate machinery fixed -- the *same* SetEncoder emits the
*same* k-dimensional z, the *same* Transport moves it -- and changes **only** how z reaches
the denoiser. Instead of the additive low-rank basis of equation (21),

    eps_hat_z = eps_hat_0 + sum_l z_l R_l(x_t, t),                    (21, the paper's model)

z enters through feature-wise affine modulation (FiLM) at every residual block, the same
locations the timestep embedding is injected. Following the FiLM implementation note, each
block gets a **dedicated** projection of z (not the timestep path):

    FiLM(H_j, z) = (1 + gamma_j(z)) * H_j + beta_j(z),   [gamma_j, beta_j] = W_j z + b_j.

That per-block projection lives in models/unet.ResBlock (built when z_film_dim is set); this
class just supplies z_film_dim=k and passes the raw coordinate down the trunk. It does NOT
fold z into the timestep embedding -- doing so would tie z to time's shared projection and
weaken the baseline. Each W_j is zero-initialised, so at z=0 (and at initialisation) FiLM is
the identity: check_backbone (which passes no z) and the z=0 control both reduce to the plain
unconditioned U-Net.
"""

from __future__ import annotations

from torch import Tensor

from models.backbone import register_backbone
from models.unet import SmallUNet, timestep_embedding


@register_backbone("film_unet")
class FiLMUNet(SmallUNet):
    """SmallUNet with a dedicated per-block FiLM projection of the task coordinate z.

    ``k`` is the coordinate dimension and must match cfg.model.k (runner/train_film.py injects
    it). forward_features keeps z optional so the DiffusionBackbone contract check -- which
    calls forward_features(x, t) with no z -- still exercises the network at z = 0.
    """

    def __init__(self, k: int = 16, **unet_kwargs) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        # Build every ResBlock with a dedicated z-FiLM projection of width k.
        super().__init__(z_film_dim=int(k), **unet_kwargs)
        self.k = int(k)

    def forward_features(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))
        if z is not None and z.dim() == 1:
            z = z.unsqueeze(0).expand(x_t.shape[0], -1)
        # temb carries time only; z modulates each block through its own W_j (in ResBlock).
        return self._trunk(x_t, temb, z if z is not None else None)
