"""Stage-1 metrics -- valid only where the true score can be computed analytically.

Section 12.1 asks for: score-field error, transported-to-oracle coordinate error, held-out
denoising loss, sample distribution error (sliced Wasserstein or MMD), and curves against k.
"""

from __future__ import annotations

import torch
from torch import Tensor

from domains.gmm2d import GMM2D
from episodes.gmm_episodes import as_batch, as_points
from models.score_model import ScoreModel


@torch.no_grad()
def score_field_error(
    model: ScoreModel,
    z: Tensor,
    gmm: GMM2D,
    n_points: int = 2048,
    t_grid: tuple[int, ...] = (50, 200, 400, 600, 800),
    span: float = 6.0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Relative error between the model score and the true score.

    Evaluation points are drawn from the **true noised distribution** x_t ~ p_t of the task,
    not from a uniform grid: what matters is accuracy where the sampler will actually go.
    """
    dev = next(model.parameters()).device
    sched = model.schedule
    per_t, num, den = {}, 0.0, 0.0

    for ti in t_grid:
        a = sched.alpha[ti].to(dev).expand(n_points)
        s = sched.sigma[ti].to(dev).expand(n_points)
        x0 = gmm.sample(n_points, generator=generator)
        xt = a.unsqueeze(1) * x0 + s.unsqueeze(1) * torch.randn(
            n_points, 2, device=dev, generator=generator)

        true_s = gmm.score_t(xt, a, s)
        tt = torch.full((n_points,), ti, device=dev, dtype=torch.long)
        pred_s = as_points(model.score(as_batch(xt), tt, z.unsqueeze(0).expand(n_points, -1)))

        e = (pred_s - true_s).pow(2).sum(1).sum()
        d = true_s.pow(2).sum(1).sum()
        per_t[f"t{ti}"] = float((e / d).sqrt())
        num, den = num + float(e), den + float(d)

    out = {"score_rel_err": (num / den) ** 0.5}
    out.update(per_t)
    return out


def sliced_wasserstein(a: Tensor, b: Tensor, n_proj: int = 256,
                       generator: torch.Generator | None = None) -> float:
    """SW_2. Compare sorted projections of the two point clouds along random directions."""
    d = a.shape[1]
    theta = torch.randn(d, n_proj, device=a.device, generator=generator)
    theta = theta / theta.norm(dim=0, keepdim=True)
    pa = (a @ theta).sort(dim=0).values
    pb = (b @ theta).sort(dim=0).values
    n = min(pa.shape[0], pb.shape[0])
    if pa.shape[0] != n:
        pa = pa[torch.linspace(0, pa.shape[0] - 1, n).long()]
    if pb.shape[0] != n:
        pb = pb[torch.linspace(0, pb.shape[0] - 1, n).long()]
    return float((pa - pb).pow(2).mean().sqrt())


def energy_mmd(a: Tensor, b: Tensor) -> float:
    """Energy distance (a form of MMD). Sensitive enough for 2D clouds and needs no kernel width."""
    def pd(x: Tensor, y: Tensor) -> Tensor:
        return torch.cdist(x, y).mean()
    return float(2 * pd(a, b) - pd(a, a) - pd(b, b))
