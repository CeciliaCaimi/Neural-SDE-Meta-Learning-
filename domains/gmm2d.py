"""The stage-1 domain: two-dimensional Gaussian mixtures. Sections 3.1 and 12.1.

    p_{y,S}(x) = Σ_{j=1..J} π_{y,j} · N(x; μ_{y,j}, Σ_{y,j}),   x ∈ R²        (11)

The entire value of this step is that it is **analytic**: Gaussian noising preserves the

    x_t = α_t·x_0 + σ_t·ε   ⇒   p_t(x) = Σ_j π_j · N(x; α_t μ_j, α_t²Σ_j + σ_t² I)

so the true score s*_t(x) = grad_x log p_t(x) has a closed form, as does the optimal noise
prediction eps* = -sigma_t s*. Three things can then be checked without a trained backbone:

  1. can the representation s_0 + B(x,t) z approximate the target score?
  2. does transport predict the change Delta_s*_y(x,t) = s*_{y,T,t}(x) - s*_{y,S,t}(x)?
  3. is a small k already enough -- what the document calls the first go/no-go

One semantic task y = one source GMM configuration. The S->T relation is a controlled
transformation recurring across tasks (rotate / translate / scale covariance / reweight),
which is what Delta_gamma must learn. Meta-test uses entirely different configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Gaussian mixture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMM2D:
    """weights (J,) | means (J,2) | covs (J,2,2). The covariances must be positive definite."""

    weights: Tensor
    means: Tensor
    covs: Tensor

    def __post_init__(self) -> None:
        J = self.weights.shape[0]
        assert self.means.shape == (J, 2), self.means.shape
        assert self.covs.shape == (J, 2, 2), self.covs.shape
        assert torch.allclose(self.weights.sum(), torch.ones((), device=self.weights.device),
                              atol=1e-5), "weights must be normalised"

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    def to(self, device) -> "GMM2D":
        return GMM2D(self.weights.to(device), self.means.to(device), self.covs.to(device))

    # ---- Sampling ----------------------------------------------------------

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        dev = self.means.device
        j = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        L = torch.linalg.cholesky(self.covs)                       # (J,2,2)
        eps = torch.randn(n, 2, device=dev, generator=generator)
        return self.means[j] + torch.einsum("nde,ne->nd", L[j], eps)

    # ---- Noised marginal ---------------------------------------------------

    def noised_params(self, alpha: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Given per-sample (alpha_t, sigma_t), return the component means and covariances of p_t.

        alpha and sigma of shape (n,) -> means (n,J,2) and covs (n,J,2,2).
        """
        a = alpha.reshape(-1, 1, 1)
        s2 = (sigma ** 2).reshape(-1, 1, 1, 1)
        eye = torch.eye(2, device=self.means.device, dtype=self.means.dtype)
        mt = a * self.means.unsqueeze(0)                            # (n,J,2)
        St = (alpha ** 2).reshape(-1, 1, 1, 1) * self.covs.unsqueeze(0) + s2 * eye
        return mt, St

    # ---- Analytic log p_t and score ---------------------------------------

    def log_prob_t(self, x: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
        """log p_t(x). x is (n,2) and alpha/sigma are (n,) -> (n,)"""
        mt, St = self.noised_params(alpha, sigma)
        diff = x.unsqueeze(1) - mt                                  # (n,J,2)
        Sinv = torch.linalg.inv(St)                                 # (n,J,2,2)
        maha = torch.einsum("njd,njde,nje->nj", diff, Sinv, diff)
        logdet = torch.logdet(St)                                   # (n,J)
        log_n = -0.5 * (maha + logdet + 2.0 * math.log(TWO_PI))
        return torch.logsumexp(self.weights.log().unsqueeze(0) + log_n, dim=1)

    def score_t(self, x: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
        """True score grad_x log p_t(x). Closed form, not autograd. x (n,2) -> (n,2)"""
        mt, St = self.noised_params(alpha, sigma)
        diff = x.unsqueeze(1) - mt                                  # (n,J,2)
        Sinv = torch.linalg.inv(St)
        maha = torch.einsum("njd,njde,nje->nj", diff, Sinv, diff)
        logdet = torch.logdet(St)
        log_n = -0.5 * (maha + logdet + 2.0 * math.log(TWO_PI))
        log_r = torch.log_softmax(self.weights.log().unsqueeze(0) + log_n, dim=1)   # (n,J)
        grad_j = -torch.einsum("njde,nje->njd", Sinv, diff)         # ∇ log N_j
        return (log_r.exp().unsqueeze(-1) * grad_j).sum(dim=1)

    def eps_star(self, x: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
        """Optimal noise prediction eps* = -sigma_t s*, equation (19). This is the learning target."""
        return -sigma.reshape(-1, 1) * self.score_t(x, alpha, sigma)


# ---------------------------------------------------------------------------
# The controlled source -> target relation. This is the thing that recurs across tasks.
# ---------------------------------------------------------------------------

RELATIONS = ("rotate", "translate", "scale_cov", "reweight")


def apply_relation(g: GMM2D, name: str, strength: float = 1.0) -> GMM2D:
    """Apply a relation to a GMM, producing its target counterpart."""
    dev, dt = g.means.device, g.means.dtype
    if name == "rotate":
        th = torch.tensor(math.pi / 3 * strength, device=dev, dtype=dt)
        R = torch.stack([torch.stack([th.cos(), -th.sin()]),
                         torch.stack([th.sin(), th.cos()])])
        return GMM2D(g.weights, g.means @ R.T, R @ g.covs @ R.T)
    if name == "translate":
        v = torch.tensor([1.5, -0.8], device=dev, dtype=dt) * strength
        return GMM2D(g.weights, g.means + v, g.covs)
    if name == "scale_cov":
        return GMM2D(g.weights, g.means, g.covs * (1.0 + 1.5 * strength))
    if name == "reweight":
        # Tilt the weights toward the first half of the components, then renormalise
        J = g.n_components
        tilt = torch.linspace(1.0 + strength, 1.0 - strength * 0.8, J, device=dev, dtype=dt)
        w = (g.weights * tilt.clamp_min(1e-3))
        return GMM2D(w / w.sum(), g.means, g.covs)
    raise ValueError(f"unknown relation '{name}'; expected one of {RELATIONS}")


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GMMTask:
    task_id: int
    relation: str
    relation_id: int
    source: GMM2D
    target: GMM2D

    def to(self, device) -> "GMMTask":
        return GMMTask(self.task_id, self.relation, self.relation_id,
                       self.source.to(device), self.target.to(device))


def random_gmm(
    n_components: int = 4,
    spread: float = 2.5,
    scale_range: tuple[float, float] = (0.15, 0.5),
    generator: torch.Generator | None = None,
    device=None,
) -> GMM2D:
    """A random source GMM: component positions, weights and covariances are all random."""
    J = n_components
    means = (torch.rand(J, 2, generator=generator, device=device) * 2 - 1) * spread
    lo, hi = scale_range
    # Lower-triangular Cholesky factor, which guarantees positive definiteness
    diag = torch.rand(J, 2, generator=generator, device=device) * (hi - lo) + lo
    off = (torch.rand(J, 1, generator=generator, device=device) * 2 - 1) * lo * 0.5
    L = torch.zeros(J, 2, 2, device=device)
    L[:, 0, 0], L[:, 1, 1] = diag[:, 0], diag[:, 1]
    L[:, 1, 0] = off[:, 0]
    covs = L @ L.transpose(-1, -2)
    logits = torch.randn(J, generator=generator, device=device) * 0.5
    return GMM2D(torch.softmax(logits, 0), means, covs)


def build_task_family(
    n_train: int = 128,
    n_test: int = 32,
    n_components: int = 4,
    relations: tuple[str, ...] = RELATIONS,
    seed: int = 12345,
    device=None,
) -> dict[str, list[GMMTask]]:
    """Generate the meta-train and meta-test task families.

    The relation set is the **same** on both sides, while the GMM configurations differ entirely.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: dict[str, list[GMMTask]] = {}
    tid = 0
    for split, n in (("train", n_train), ("test", n_test)):
        tasks = []
        for _ in range(n):
            src = random_gmm(n_components, generator=g)
            rel_i = int(torch.randint(len(relations), (1,), generator=g).item())
            rel = relations[rel_i]
            tgt = apply_relation(src, rel)
            t = GMMTask(tid, rel, rel_i, src, tgt)
            tasks.append(t if device is None else t.to(device))
            tid += 1
        out[split] = tasks
    return out


# ---------------------------------------------------------------------------
# A related task family: one common template plus small perturbations
# ---------------------------------------------------------------------------
# Section 1.2 says "within a **related family** of tasks": the hypothesis concerns the low
# dimensionality of the **change** between related distributions, not of the distributions
# themselves. random_gmm tasks are mutually unrelated, so s_0 learns nothing shared and z
# must encode a whole task alone. Here relatedness is tunable: 0 makes all tasks identical.

def perturb_gmm(
    template: GMM2D,
    strength: float,
    spread: float = 2.5,
    generator: torch.Generator | None = None,
) -> GMM2D:
    """Perturb a task within the template neighbourhood. strength=0 returns the template itself."""
    J = template.n_components
    dev, dt = template.means.device, template.means.dtype
    g = generator

    means = template.means + strength * spread * torch.randn(
        J, 2, generator=g, device=dev, dtype=dt)

    # Perturb the covariance on a log scale via its Cholesky factor, preserving definiteness
    L = torch.linalg.cholesky(template.covs)
    L = L * torch.exp(strength * torch.randn(J, 1, 1, generator=g, device=dev, dtype=dt))
    covs = L @ L.transpose(-1, -2)

    logits = template.weights.log() + strength * torch.randn(
        J, generator=g, device=dev, dtype=dt)
    return GMM2D(torch.softmax(logits, 0), means, covs)


def build_related_task_family(
    n_train: int = 128,
    n_test: int = 32,
    n_components: int = 4,
    perturb: float = 0.25,
    relations: tuple[str, ...] = RELATIONS,
    seed: int = 12345,
    device=None,
) -> dict[str, list[GMMTask]]:
    """Every task is a perturbed version of **one template**.

    Train and test share the template but use **different perturbations**, so meta-test tasks
    stay unseen: the configuration is unseen, not the family. Reusable relations need exactly this.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    template = random_gmm(n_components, generator=g)

    out: dict[str, list[GMMTask]] = {}
    tid = 0
    for split, n in (("train", n_train), ("test", n_test)):
        tasks = []
        for _ in range(n):
            src = perturb_gmm(template, perturb, generator=g)
            rel_i = int(torch.randint(len(relations), (1,), generator=g).item())
            t = GMMTask(tid, relations[rel_i], rel_i, src, apply_relation(src, relations[rel_i]))
            tasks.append(t if device is None else t.to(device))
            tid += 1
        out[split] = tasks
    return out
