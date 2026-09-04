"""The foundation of stage 1: the analytic score must be correct.

    python -m tests.test_gmm_analytic

If this is wrong, every later conclusion about "the model approximates the true score" is
false. score_t is therefore cross-validated by **three mutually independent** methods:

  1. autograd through log_prob_t -- score_t is a separately written closed form that never
     calls log_prob_t, so this is a genuinely independent check
  2. finite differences
  3. Monte Carlo: sample x_t = alpha x_0 + sigma eps and compare empirical to analytic moments

Also: does p_t integrate to 1, is eps* = -sigma s* consistent, do the relations give valid GMMs.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusion.schedule import NoiseSchedule                        # noqa: E402
from domains.gmm2d import (                                          # noqa: E402
    RELATIONS, apply_relation, build_task_family, random_gmm,
)

RULE = "─" * 78
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ——  {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)          # double precision for the analytic checks
    print(RULE)
    print("GMM analytic score verification")
    print(RULE)

    g = torch.Generator().manual_seed(7)
    gmm = random_gmm(n_components=4, generator=g)
    sched = NoiseSchedule(1000, "cosine").double()

    n = 512
    x = (torch.rand(n, 2, generator=g) * 2 - 1) * 6
    t = torch.randint(0, 1000, (n,), generator=g)
    alpha, sigma = sched.alpha[t].double(), sched.sigma[t].double()

    # ---- 1. Autograd ----
    print("\n[1] cross-check against autograd of log p_t")
    xg = x.clone().requires_grad_(True)
    lp = gmm.log_prob_t(xg, alpha, sigma)
    (grad,) = torch.autograd.grad(lp.sum(), xg)
    s = gmm.score_t(x, alpha, sigma)
    err = (grad - s).abs().max().item()
    rel = ((grad - s).norm() / grad.norm()).item()
    record("score_t == grad_x log p_t", err < 1e-8, f"max absolute deviation {err:.2e}, relative {rel:.2e}")

    # ---- 2. Finite differences ----
    print("\n[2] finite differences")
    h = 1e-5
    fd = torch.zeros_like(x)
    for d in range(2):
        e = torch.zeros_like(x); e[:, d] = h
        fd[:, d] = (gmm.log_prob_t(x + e, alpha, sigma) - gmm.log_prob_t(x - e, alpha, sigma)) / (2 * h)
    rel_fd = ((fd - s).norm() / s.norm()).item()
    record("score_t == finite differences", rel_fd < 1e-6, f"relative deviation {rel_fd:.2e}")

    # ---- 3. Monte Carlo moments of the noised marginal ----
    print("\n[3] Monte Carlo check of the noised marginal")
    ti = 400
    a1, s1 = sched.alpha[ti].double(), sched.sigma[ti].double()
    m = 400_000
    x0 = gmm.sample(m, generator=g)
    xt = a1 * x0 + s1 * torch.randn(m, 2, generator=g)
    mt, St = gmm.noised_params(a1.reshape(1), s1.reshape(1))
    w = gmm.weights
    mean_an = (w[:, None] * mt[0]).sum(0)
    # covariance of a mixture = E[Sigma_j] + Cov(mu_j)
    cov_an = (w[:, None, None] * St[0]).sum(0) + \
             (w[:, None, None] * torch.einsum("jd,je->jde", mt[0] - mean_an, mt[0] - mean_an)).sum(0)
    dm = (xt.mean(0) - mean_an).norm().item()
    dc = (torch.cov(xt.T) - cov_an).norm().item() / cov_an.norm().item()
    record("noised sample mean matches the analytic mean", dm < 0.02, f"deviation {dm:.4f}  (m={m:,})")
    record("noised sample covariance matches the analytic one", dc < 0.02, f"relative deviation {dc:.4f}")

    # ---- 4. Normalisation ----
    print("\n[4] normalisation of p_t (grid integration)")
    lim, gn = 12.0, 600
    ax = torch.linspace(-lim, lim, gn)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    for ti in (0, 400, 900):
        a = sched.alpha[ti].double().expand(pts.shape[0])
        sg = sched.sigma[ti].double().expand(pts.shape[0])
        dens = gmm.log_prob_t(pts, a, sg).exp()
        integral = dens.sum().item() * (2 * lim / (gn - 1)) ** 2
        record(f"∫p_t dx = 1  (t={ti})", abs(integral - 1) < 5e-3, f"{integral:.6f}")

    # ---- 5. Consistency of eps* ----
    print("\n[5] eps* = -sigma_t s*  (equation 19)")
    e_star = gmm.eps_star(x, alpha, sigma)
    record("eps_star agrees with score_t",
           torch.allclose(e_star, -sigma.reshape(-1, 1) * s, atol=1e-12))
    # High-noise end: x_t is about sigma eps, so eps* is about x_t / sigma
    hi = torch.full((n,), 999)
    a_hi, s_hi = sched.alpha[hi].double(), sched.sigma[hi].double()
    xt_hi = a_hi.reshape(-1, 1) * gmm.sample(n, generator=g) + s_hi.reshape(-1, 1) * torch.randn(n, 2, generator=g)
    e_hi = gmm.eps_star(xt_hi, a_hi, s_hi)
    approx = xt_hi / s_hi.reshape(-1, 1)
    rel_hi = ((e_hi - approx).norm() / approx.norm()).item()
    record("eps* -> x_t / sigma_t as t -> T", rel_hi < 0.05, f"relative deviation {rel_hi:.3f}")

    # ---- 6. Relations ----
    print("\n[6] the four source-to-target relations")
    ok_all, details = True, []
    for name in RELATIONS:
        tgt = apply_relation(gmm, name)
        pd = bool((torch.linalg.eigvalsh(tgt.covs) > 0).all())
        norm = bool(torch.allclose(tgt.weights.sum(), torch.tensor(1.0, dtype=torch.float64)))
        moved = (tgt.means - gmm.means).norm().item() + (tgt.covs - gmm.covs).norm().item() \
                + (tgt.weights - gmm.weights).norm().item()
        ok_all &= pd and norm and moved > 1e-6
        details.append(f"{name}(Δ={moved:.2f})")
    record("relations produce valid, non-trivial GMMs", ok_all, " ".join(details))

    # Does the relation actually change the score field? Delta_s*_y(x,t) must not be 0
    tgt = apply_relation(gmm, "rotate")
    ds = (tgt.score_t(x, alpha, sigma) - s).norm() / s.norm()
    record("Delta_s* = s*_T - s*_S is nonzero", ds.item() > 0.1, f"relative magnitude {ds.item():.3f}")

    # ---- 7. Task family ----
    print("\n[7] task family split")
    fam = build_task_family(n_train=64, n_test=16, seed=1)
    tr_means = torch.stack([t.source.means for t in fam["train"]])
    te_means = torch.stack([t.source.means for t in fam["test"]])
    # Semantic tasks (GMM configurations) must be entirely different
    dup = sum(1 for a in te_means for b in tr_means if torch.allclose(a, b))
    record("no meta-test GMM configuration appears in train", dup == 0,
           f"train {len(fam['train'])} · test {len(fam['test'])}")
    rel_tr = {t.relation for t in fam["train"]}
    rel_te = {t.relation for t in fam["test"]}
    record("the relation set is identical on both sides (relations must be reusable)", rel_te <= rel_tr, f"{sorted(rel_tr)}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(RULE)
    print(f"{len(results) - n_fail} / {len(results)} checks passed" + ("" if not n_fail else f"  --  {n_fail} failed"))
    print(RULE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
