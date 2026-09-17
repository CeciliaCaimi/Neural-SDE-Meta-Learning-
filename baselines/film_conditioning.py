"""E15: the generic-latent-conditioning comparison arm.

FiLMScoreModel keeps everything ScoreModel exposes -- eps_hat, eps_hat_many, score,
split_eps, basis_usage -- so the training loop, the diagnostics and the samplers reach it
through the identical interface. What changes is *where z enters*:

  - ScoreModel (equation 21): eps_hat_z = eps_hat_0 + sum_l z_l R_l(x_t, t).
    z is an additive, low-rank, per-direction perturbation of the score.
  - FiLMScoreModel: z is fed to the backbone (models/film_unet.py), which folds it into
    every residual block's scale-shift. eps_hat_z = base_head(features(x_t, t, z)).
    There is **no basis term** -- this is the "just condition the U-Net on the task vector"
    baseline that the paper's structured basis is measured against.

The comparison is matched: same SetEncoder, same Transport, same k, same base head, same
loss. Only the conditioning mechanism differs, which is the whole point of the arm.

Requires a backbone whose forward_features accepts z (FiLMUNet). Centring (E13) is orthogonal
and left off for this arm, but _prepare_z is still used so z passes through h_eta / centring
identically to the main model.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

from models.score_model import ScoreModel, register_score_model


@register_score_model("film")
class FiLMScoreModel(ScoreModel):
    """z conditions the backbone (FiLM) instead of an additive score basis."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # No additive basis in this arm. The head is inherited from ScoreModel for
        # state-dict / parameter-group uniformity, but it is never evaluated (the overrides
        # below never call self.basis) and is frozen so it draws no gradient and cannot
        # smuggle in the very mechanism this baseline exists to exclude.
        for p in self.basis_head.parameters():
            p.requires_grad_(False)
        # Fail loudly rather than conditioning on nothing: a backbone whose
        # forward_features ignores z would train an arm that answers a different
        # question while every diagnostic still reads plausibly.
        if not getattr(self.backbone, "accepts_z", False):
            raise TypeError(
                f"{type(self.backbone).__name__} does not take z in forward_features; "
                "the FiLM arm needs models/film_unet.py")

    # ---- z now reaches the denoiser through the backbone, not a head ----

    def _features_cond(self, x_t: Tensor, t: Tensor, z: Tensor | None) -> Tensor:
        """Backbone features conditioned on z. z passes through _prepare_z first, so
        centring and h_eta apply exactly as they do for the main model."""
        dtype = next(self.parameters()).dtype
        zz = self._prepare_z(z, x_t.shape[0], x_t.device, dtype)
        return self.backbone.forward_features(x_t, t, zz)

    def eps_hat(self, x_t: Tensor, t: Tensor, z: Tensor | None = None) -> Tensor:
        """No equation (21) sum here: z-conditioned features straight through the base head."""
        return self.eps_base(x_t, t, self._features_cond(x_t, t, z))

    def eps_hat_many(self, x_t: Tensor, t: Tensor, zs: Sequence[Tensor | None]) -> list[Tensor]:
        """Each z needs its own backbone pass -- FiLM makes the features depend on z, so the
        one-pass sharing that the additive basis allows does not apply here."""
        return [self.eps_hat(x_t, t, z) for z in zs]

    def split_eps(
        self, x_t: Tensor, t: Tensor, z: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """(base, residual) with base = the z = 0 prediction and residual = the change z
        induces. Keeps r_basis / basis_usage meaningful as "how much does z move the output",
        the FiLM analogue of the basis-usage ratio."""
        base = self.eps_hat(x_t, t, None)
        return base, self.eps_hat(x_t, t, z) - base
