# evaluation/compositional_metrics.py
#
# Item 6, phase 1b: a MechanismTheta-aware version of
# adaptation/gated_finetuning_regularized.compute_mechanism_error, so
# drift_error/diffusion_error can be scored against the item-5 factorial
# benchmark's ground truth (mechanism_drift/mechanism_diffusion from
# data_gen/mechanism_library.py) instead of the legacy basis-coefficient
# ground truth (drift_true/sigma_diag_true from
# sde_basis/parameterised_sde.py, which only knows how to read a `Theta`,
# not a `MechanismTheta`).
#
# This duplicates compute_mechanism_error's body rather than editing it or
# adding a branch to it -- same rationale as
# data_gen/simulate_true_sde.py's simulate_batch_generic: the existing
# function is depended on by items 1-4's already-queued real-GPU
# comparisons, so it is not touched. The normalised/raw coordinate handling
# (rescale drift/diffusion predictions back to raw simulator units by the
# target scaler's per-dimension std, when a scaler is given) is copied
# exactly.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from data_gen.mechanism_library import MechanismTheta, mechanism_diffusion, mechanism_drift


def compute_mechanism_error_factorial(
    sde,
    theta: MechanismTheta,
    zf: torch.Tensor,
    zg: torch.Tensor,
    query_in: torch.Tensor,
    query_orig: torch.Tensor,
    target_scaler: Optional[object],
) -> Tuple[float, float]:
    """Drift/diffusion identification error against a MechanismTheta's true
    mechanism_drift/mechanism_diffusion, evaluated pointwise at every
    observed query state.

    Args mirror compute_mechanism_error exactly:
        theta: ground-truth MechanismTheta for this task.
        zf, zg: adapted drift/diffusion latents, shape (1, z_dim).
        query_in: (N_query, T, D) query trajectory, in the (possibly
            normalized) coordinate frame the SDE nets were trained in.
        query_orig: (N_query, T, D) query trajectory in original simulator
            units -- this is the frame mechanism_drift/mechanism_diffusion
            operate in.
        target_scaler: the StandardScaler used to go from query_orig to
            query_in, or None if the model operates on raw data.
    """
    B, T, D = query_orig.shape
    x_raw = query_orig.reshape(-1, D)
    x_in = query_in.reshape(-1, D)
    n = x_in.shape[0]

    zf_exp = zf.expand(n, -1)
    zg_exp = zg.expand(n, -1)
    t_dummy = torch.zeros(n, device=x_in.device)

    with torch.no_grad():
        drift_pred = sde.f(t_dummy, x_in, zf_exp)                       # (n, D)
        diff_pred_mat = sde.g(t_dummy, x_in, zg_exp)                    # (n, D, D)
        diff_pred = torch.diagonal(diff_pred_mat, dim1=-2, dim2=-1)     # (n, D)

    if target_scaler is not None:
        scale = torch.as_tensor(
            target_scaler.scale_, dtype=drift_pred.dtype, device=drift_pred.device
        )
        drift_pred = drift_pred * scale
        diff_pred = diff_pred * scale

    drift_target = mechanism_drift(x_raw, theta)          # (n, D)
    diff_target = mechanism_diffusion(x_raw, theta)        # (n, D)

    drift_error = F.mse_loss(drift_pred, drift_target).item()
    diffusion_error = F.mse_loss(diff_pred, diff_target).item()
    return drift_error, diffusion_error
