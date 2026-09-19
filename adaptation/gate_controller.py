# adaptation/gate_controller.py
#
# Item 3: a learned, drift/diffusion-specific adaptation controller.
#
# This is an ADDITIONAL selectable gating mode alongside the existing scalar
# gate in adaptation/gated_finetuning_regularized.py (g = sigmoid(alpha *
# (tau - d_norm))), not a replacement. The scalar gate blends two full
# *simulation outputs* (a "smart" rollout using the adapted latents and a
# "safe" rollout using zeroed latents) with one scalar weight shared by both
# z_f and z_g. This controller instead predicts two independent weights
# (a_f, a_g) and blends *in latent space*, before simulation:
#
#     z_f' = z_f(0) + a_f * (z_f_adapted - z_f(0))
#     z_g' = z_g(0) + a_g * (z_g_adapted - z_g(0))
#
# See adaptation/gated_finetuning_regularized.py's gated_inference() for how
# the two modes coexist, and training/train_controller.py for how (a_f, a_g)
# are meta-trained.

import torch
import torch.nn as nn

from models.mlp import MLP


class LearnedGateController(nn.Module):
    """Small MLP mapping per-task adaptation signals to (a_f, a_g) in [0,1]^2.

    Input (concatenated by build_controller_input, total dim = 2 * z_dim + 4):
      - zf_init, zg_init: pooled pre-adaptation encoder latents (z_dim each)
        -- the same support-set mean-pooled representation gated_inference
        already computes, reused as-is rather than adding a new encoder path.
      - drift_residual, diffusion_residual: item-1's drift_error/
        diffusion_error-style mechanism residuals (1 each), evaluated on the
        support set *pre-adaptation* (i.e. with zf_init/zg_init).
      - grad_norm_zf, grad_norm_zg: gradient norm of the support-set
        adaptation loss w.r.t. z_f and z_g at the initial point (1 each),
        captured as a byproduct of the first step of the existing Adam
        fine-tuning loop in adapt_model().

    Output: 2 logits -> sigmoid -> (a_f, a_g), each in [0, 1].
    """

    def __init__(self, z_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        input_dim = 2 * z_dim + 4
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=2,
            use_layernorm=False,
        )

    def forward(self, controller_input: torch.Tensor) -> torch.Tensor:
        """controller_input: (batch, 2*z_dim+4) -> (batch, 2) = (a_f, a_g)."""
        logits = self.net(controller_input)
        return torch.sigmoid(logits)


def build_controller_input(
    zf_init: torch.Tensor,
    zg_init: torch.Tensor,
    drift_residual: float,
    diffusion_residual: float,
    grad_norm_zf: float,
    grad_norm_zg: float,
) -> torch.Tensor:
    """Concatenate the controller's exact inputs into one (1, 2*z_dim+4) tensor.

    zf_init/zg_init are expected to already carry no gradient history (they
    come from a `with torch.no_grad()` encoder pass in the caller); the
    explicit .detach() here just makes that guarantee visible at the call
    site rather than relying on the caller having done it correctly.
    drift_residual/diffusion_residual/grad_norm_zf/grad_norm_zg are plain
    Python floats (already detached scalar byproducts of upstream no_grad
    computations), used purely as conditioning signal for the controller --
    nothing upstream of them needs to be differentiable.
    """
    device = zf_init.device
    dtype = zf_init.dtype
    scalars = torch.tensor(
        [drift_residual, diffusion_residual, grad_norm_zf, grad_norm_zg],
        dtype=dtype, device=device,
    ).unsqueeze(0)  # (1, 4)
    return torch.cat([zf_init.detach(), zg_init.detach(), scalars], dim=-1)
