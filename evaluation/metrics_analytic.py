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


def kid(a: Tensor, b: Tensor, subset_size: int = 100, n_subsets: int = 100,
        generator: torch.Generator | None = None) -> tuple[float, float]:
    """Kernel Inception Distance: the unbiased MMD^2 under the cubic polynomial kernel.

        k(x, y) = (x.y / d + 1) ^ 3

    Returns (mean, 95 % half-width) over ``n_subsets`` random subsets of ``subset_size``
    drawn from each side, which is how the statistic is normally reported: the unbiased
    estimator has no closed-form variance, and subset averaging both bounds the cost and
    supplies the spread.

    ``a`` and ``b`` are feature rows, not images. This project has no InceptionV3, so the
    features come from evaluation.instruments.TinyCNN -- the same instrument whose held-out
    accuracy is quoted beside every semantic verdict. That makes this KID's *estimator*
    standard and its *feature space* local, so the number is comparable between arms
    measured here and not comparable with a published KID. Say so wherever it is reported.
    """
    m = min(a.shape[0], b.shape[0])
    if subset_size > m:
        subset_size = m
    if subset_size < 2:
        raise ValueError(f"need at least 2 samples per side, got {m}")
    d = a.shape[1]

    def poly(x: Tensor, y: Tensor) -> Tensor:
        return (x @ y.t() / d + 1.0).pow(3)

    vals = []
    n = subset_size
    eye = torch.eye(n, dtype=torch.bool, device=a.device)
    for _ in range(n_subsets):
        ia = torch.randperm(a.shape[0], device=a.device, generator=generator)[:n]
        ib = torch.randperm(b.shape[0], device=b.device, generator=generator)[:n]
        xa, xb = a[ia], b[ib]
        kaa = poly(xa, xa).masked_fill(eye, 0.0).sum() / (n * (n - 1))
        kbb = poly(xb, xb).masked_fill(eye, 0.0).sum() / (n * (n - 1))
        kab = poly(xa, xb).mean()
        vals.append(float(kaa + kbb - 2 * kab))
    t = torch.tensor(vals)
    if len(vals) < 2:
        return float(t.mean()), float("nan")
    return float(t.mean()), float(1.96 * t.std(unbiased=True) / (len(vals) ** 0.5))
