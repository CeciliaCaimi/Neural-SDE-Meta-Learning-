# data_gen/mechanism_library.py
#
# Item 5: a factorial drift/diffusion mechanism library, independent of and
# additional to the existing basis-coefficient scheme in
# sde_basis/basis_functions.py + sde_basis/parameterised_sde.py.
#
# --- Why this exists (item 5, step 0 investigation) ---------------------
#
# The existing task generator (data_gen/generate_meta_params.py) samples a
# `Theta(theta_b, theta_sigma)` whose drift and diffusion are BOTH linear
# combinations of one fixed, shared basis per state dimension:
#   drift:     b(x)     = sum_k theta_b[d,k]     * phi_b[k](x)      (8 terms:
#              1, x, x^2, x^3, sin(x), cos(x), tanh(x), x^4 -- see
#              sde_basis/basis_functions.py:drift_basis)
#   diffusion: sigma(x) = softplus(sum_k theta_sigma[d,k] * phi_s[k](x)) + eps
#              (3 terms: 1, |x|, sqrt(|x|) -- diffusion_basis)
# theta_b and theta_sigma ARE sampled independently (sample_theta draws two
# separate torch.randn calls), so at the *coefficient* level drift and
# diffusion are already decoupled -- see onboarding_summary.md. The actual
# coupling item 5 is meant to break is two levels up:
#
#   1. Every task, in every split and every regime, is drawn from the SAME
#      fixed 8-term / 3-term functional family. There is no notion of
#      "qualitatively different dynamics" (stable vs oscillatory vs
#      multistable vs ...) -- only different coefficients on one universal
#      polynomial+trig basis. A model can partially "cheat" by learning
#      properties of that one family rather than a mechanism-general notion
#      of drift/diffusion.
#   2. The named shift regimes (cfg.theta_dist.shift_regimes, consumed by
#      generate_meta_params.generate_all_meta_params) each specify a SINGLE
#      joint (drift_shift, diffusion_shift, std_scale) triple. Even
#      "drift_only"/"diffusion_only" are just two special-cased slices of
#      one shared regime table -- drift and diffusion severity are not
#      independently, continuously, jointly sampled at scale; there are only
#      9 hand-picked (drift_shift, diffusion_shift) pairs total.
#
# This module fixes both: it defines many qualitatively distinct drift
# mechanisms and diffusion mechanisms (each with its own coefficient prior),
# then data_gen/generate_factorial_benchmark.py draws a drift instance and a
# diffusion instance completely independently and crosses them, giving
# hundreds-to-thousands of (drift mechanism, drift coeffs, diffusion
# mechanism, diffusion coeffs) combinations instead of 9 regimes over 1
# family.
#
# --- Interface parity / divergence with sde_basis -----------------------
#
# Downstream consumers (e.g. adaptation/gated_finetuning_regularized.py's
# compute_mechanism_error) call the *true* drift/diffusion as plain
# functions of state and a task object:
#     drift_target = drift_true(x_raw, theta)          # (..., d) -> (..., d)
#     diff_target  = sigma_diag_true(x_raw, theta)      # (..., d) -> (..., d)
# where `theta` carries theta_b/theta_sigma and drift_true/sigma_diag_true
# dispatch through the fixed basis functions above.
#
# This module preserves that *shape contract* exactly -- mechanism_drift(x,
# theta) and mechanism_diffusion(x, theta) both take x of shape (..., d) and
# a task object, and return (..., d), with drift depending only on each
# state dimension's own coordinate (the same "diagonal-only" assumption
# drift_true documents) so x_dim and the correlated-noise-via-Cholesky-L
# machinery in data_gen/simulate_true_sde.py need no changes. It deliberately
# does NOT preserve the fixed-basis *implementation* (a dot product against
# theta_b/theta_sigma) -- MechanismTheta stores a mechanism *name* plus
# per-dimension coefficients specific to that mechanism's own functional
# form, dispatched via DRIFT_BY_NAME/DIFFUSION_BY_NAME instead of a basis
# matrix. Reusing the fixed 8/3-term basis would defeat the point of item 5
# (qualitatively different families), since e.g. a saturating rational
# function 1/(1+|x|) or a genuine multiplicative-in-|x| diffusion are not
# expressible as that basis's linear combination.
#
# Existing consumers that hardcode `from sde_basis.parameterised_sde import
# drift_true, sigma_diag_true` and a `Theta` argument are therefore NOT
# wired up to consume MechanismTheta tasks directly -- that would require
# editing that eval code, which is out of scope for item 5 (no NN/eval code
# changes). See the note at the bottom of generate_factorial_benchmark.py
# for how a future consumer should be adapted.

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch

Coeffs = Dict[str, torch.Tensor]  # each value: shape (d,), one draw per state dim


def _u(d: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    """Per-dimension uniform draw on [low, high)."""
    return torch.rand(d, generator=generator) * (high - low) + low


def _n(d: int, std: float, generator: torch.Generator) -> torch.Tensor:
    """Per-dimension zero-mean Gaussian draw."""
    return torch.randn(d, generator=generator) * std


@dataclass
class DriftMechanism:
    name: str
    category: str
    sampler: Callable[[int, torch.Generator], Coeffs]
    fn: Callable[[torch.Tensor, Coeffs], torch.Tensor]


@dataclass
class DiffusionMechanism:
    name: str
    category: str
    sampler: Callable[[int, torch.Generator], Coeffs]
    fn: Callable[[torch.Tensor, Coeffs], torch.Tensor]


# =====================================================================
# Drift mechanisms -- 5 categories x 2-3 mechanisms = 14 total.
# =====================================================================

DRIFT_MECHANISMS: List[DriftMechanism] = [
    # ---- Category: stable linear systems ----
    DriftMechanism(
        "ou_linear", "stable_linear",
        sampler=lambda d, g: {"a": _u(d, 0.3, 2.0, g), "mu": _n(d, 0.3, g)},
        fn=lambda x, c: -c["a"] * (x - c["mu"]),
    ),
    DriftMechanism(
        "piecewise_linear_asym", "stable_linear",
        sampler=lambda d, g: {"a_pos": _u(d, 0.3, 2.0, g), "a_neg": _u(d, 0.3, 2.0, g)},
        fn=lambda x, c: -torch.where(x >= 0, c["a_pos"] * x, c["a_neg"] * x),
    ),

    # ---- Category: oscillatory systems (periodic-drift potentials) ----
    # NOTE: drift here is diagonal-per-dimension (see module docstring / the
    # existing drift_true convention), so a true limit-cycle oscillator
    # (which needs a position/velocity pair coupled off-diagonal) isn't
    # representable. "Oscillatory" is implemented as an overdamped periodic
    # potential (b(x) = -grad of a periodic potential), which is the
    # standard SDE-literature stand-in for oscillatory drift under a
    # diagonal-only assumption -- it produces genuinely periodic (infinitely
    # repeating) equilibria, distinguishing it from the finite-well
    # "multistable" category below.
    DriftMechanism(
        "phase_oscillator", "oscillatory",
        sampler=lambda d, g: {
            "a": _u(d, 0.3, 1.5, g), "w": _u(d, 1.0, 3.0, g),
            "phase": _u(d, 0.0, 6.283185307, g),
        },
        fn=lambda x, c: -c["a"] * torch.sin(c["w"] * (x - c["phase"])),
    ),
    DriftMechanism(
        "damped_periodic", "oscillatory",
        sampler=lambda d, g: {
            "a": _u(d, 0.2, 1.0, g), "b": _u(d, 0.2, 1.0, g), "w": _u(d, 1.0, 3.0, g),
        },
        fn=lambda x, c: -c["a"] * x + c["b"] * torch.sin(c["w"] * x),
    ),
    DriftMechanism(
        "dual_frequency_periodic", "oscillatory",
        sampler=lambda d, g: {
            "a1": _u(d, 0.2, 1.0, g), "w1": _u(d, 1.0, 2.0, g),
            "a2": _u(d, 0.1, 0.6, g), "w2": _u(d, 2.0, 4.0, g),
        },
        fn=lambda x, c: -c["a1"] * torch.sin(c["w1"] * x) - c["a2"] * torch.sin(c["w2"] * x),
    ),

    # ---- Category: multistable systems (finite polynomial-well potentials) ----
    DriftMechanism(
        "double_well", "multistable",
        sampler=lambda d, g: {"a": _u(d, 0.5, 2.0, g), "c": _u(d, 0.5, 1.5, g)},
        fn=lambda x, c: c["a"] * x - c["c"] * x ** 3,
    ),
    DriftMechanism(
        "asym_double_well", "multistable",
        sampler=lambda d, g: {
            "a": _u(d, 0.5, 2.0, g), "c": _u(d, 0.5, 1.5, g), "tilt": _u(d, -0.3, 0.3, g),
        },
        fn=lambda x, c: c["a"] * x - c["c"] * x ** 3 + c["tilt"],
    ),
    DriftMechanism(
        "steep_double_well", "multistable",
        sampler=lambda d, g: {
            "a": _u(d, 0.5, 2.0, g), "c": _u(d, 0.5, 1.5, g), "e": _u(d, 0.02, 0.15, g),
        },
        # Same bistable topology as double_well but with a quintic (rather
        # than purely cubic) large-|x| damping term -- a differently-shaped
        # potential well (flatter basin, steeper/faster-saturating walls)
        # and a different numerical-stability profile.
        fn=lambda x, c: c["a"] * x - c["c"] * x ** 3 - c["e"] * x ** 5,
    ),

    # ---- Category: nonlinear bounded systems (saturating restoring force) ----
    DriftMechanism(
        "tanh_saturating", "nonlinear_bounded",
        sampler=lambda d, g: {"a": _u(d, 0.3, 1.5, g), "w": _u(d, 0.5, 2.0, g)},
        fn=lambda x, c: -c["a"] * torch.tanh(c["w"] * x),
    ),
    DriftMechanism(
        "atan_saturating", "nonlinear_bounded",
        sampler=lambda d, g: {"a": _u(d, 0.3, 1.5, g), "w": _u(d, 0.5, 2.0, g)},
        fn=lambda x, c: -c["a"] * torch.atan(c["w"] * x),
    ),
    DriftMechanism(
        "rational_saturating", "nonlinear_bounded",
        sampler=lambda d, g: {"a": _u(d, 0.3, 1.5, g)},
        fn=lambda x, c: -c["a"] * x / (1.0 + torch.abs(x)),
    ),

    # ---- Category: weakly unstable systems ----
    # Locally repelling near the origin. marginal_linear_unstable has NO
    # containment term at all (relies on max_state_abs / short horizon --
    # expected to trigger the stability-skip guard at a non-trivial rate,
    # by design); the other two are locally unstable at 0 but globally
    # bounded via a higher-order term (supercritical-pitchfork-like).
    DriftMechanism(
        "marginal_linear_unstable", "weakly_unstable",
        sampler=lambda d, g: {"eps": _u(d, 0.05, 0.4, g)},
        fn=lambda x, c: c["eps"] * x,
    ),
    DriftMechanism(
        "weak_pitchfork", "weakly_unstable",
        sampler=lambda d, g: {"eps": _u(d, 0.02, 0.2, g), "c": _u(d, 0.3, 1.0, g)},
        fn=lambda x, c: c["eps"] * x - c["c"] * x ** 3,
    ),
    DriftMechanism(
        "asym_weak_instability", "weakly_unstable",
        sampler=lambda d, g: {
            "eps": _u(d, 0.02, 0.2, g), "q": _u(d, 0.05, 0.3, g), "c": _u(d, 0.2, 0.8, g),
        },
        fn=lambda x, c: c["eps"] * x + c["q"] * x ** 2 - c["c"] * x ** 4,
    ),
]

DRIFT_CATEGORIES: Tuple[str, ...] = (
    "stable_linear", "oscillatory", "multistable", "nonlinear_bounded", "weakly_unstable",
)

DRIFT_BY_NAME: Dict[str, DriftMechanism] = {m.name: m for m in DRIFT_MECHANISMS}


# =====================================================================
# Diffusion mechanisms -- 3 categories x 2-3 mechanisms = 8 total.
# All functional forms are non-negative by construction (abs/sqrt/square of
# positively-sampled coefficients); mechanism_diffusion() still applies a
# small positive floor defensively, matching BasisConfig.diffusion_eps.
# =====================================================================

DIFFUSION_MECHANISMS: List[DiffusionMechanism] = [
    # ---- Category: approximately constant diffusion ----
    DiffusionMechanism(
        "const_diffusion", "approx_constant",
        sampler=lambda d, g: {"c": _u(d, 0.1, 0.6, g)},
        fn=lambda x, c: c["c"] * torch.ones_like(x),
    ),
    DiffusionMechanism(
        "near_constant_weak_state", "approx_constant",
        sampler=lambda d, g: {"c": _u(d, 0.1, 0.5, g), "eps": _u(d, 0.0, 0.05, g)},
        fn=lambda x, c: c["c"] + c["eps"] * torch.abs(x),
    ),

    # ---- Category: state-dependent diffusion ----
    DiffusionMechanism(
        "cir_sqrt", "state_dependent",
        sampler=lambda d, g: {"c": _u(d, 0.2, 0.8, g)},
        fn=lambda x, c: c["c"] * torch.sqrt(torch.abs(x) + 1e-6),
    ),
    DiffusionMechanism(
        "quadratic_state", "state_dependent",
        sampler=lambda d, g: {"c0": _u(d, 0.05, 0.3, g), "c1": _u(d, 0.05, 0.3, g)},
        fn=lambda x, c: c["c0"] + c["c1"] * x ** 2,
    ),
    DiffusionMechanism(
        "abs_linear_state", "state_dependent",
        sampler=lambda d, g: {"c0": _u(d, 0.05, 0.3, g), "c1": _u(d, 0.2, 0.8, g)},
        fn=lambda x, c: c["c0"] + c["c1"] * torch.abs(x),
    ),

    # ---- Category: multiplicative diffusion (vol proportional to level) ----
    DiffusionMechanism(
        "gbm_multiplicative", "multiplicative",
        sampler=lambda d, g: {"c": _u(d, 0.2, 0.8, g)},
        fn=lambda x, c: c["c"] * torch.abs(x),
    ),
    DiffusionMechanism(
        "gbm_with_floor", "multiplicative",
        sampler=lambda d, g: {"c": _u(d, 0.2, 0.8, g), "floor": _u(d, 0.02, 0.1, g)},
        fn=lambda x, c: c["c"] * torch.abs(x) + c["floor"],
    ),
    DiffusionMechanism(
        "bounded_multiplicative", "multiplicative",
        sampler=lambda d, g: {"c": _u(d, 0.3, 1.0, g), "floor": _u(d, 0.02, 0.1, g)},
        fn=lambda x, c: c["c"] * torch.abs(x) / (1.0 + torch.abs(x)) + c["floor"],
    ),
]

DIFFUSION_CATEGORIES: Tuple[str, ...] = ("approx_constant", "state_dependent", "multiplicative")

DIFFUSION_BY_NAME: Dict[str, DiffusionMechanism] = {m.name: m for m in DIFFUSION_MECHANISMS}


# =====================================================================
# Task representation + factorial sampling
# =====================================================================

@dataclass
class MechanismTheta:
    """
    A single factorial task: one drift mechanism instance x one diffusion
    mechanism instance, sampled completely independently of each other.

    Mirrors sde_basis.parameters.Theta's role (a container of "true" SDE
    parameters plus an id, with a `.to(device)` method) but stores mechanism
    names + per-mechanism coefficient dicts instead of a shared theta_b /
    theta_sigma basis matrix -- see the module docstring for why.
    """
    drift_mechanism: str
    drift_category: str
    drift_coeffs: Coeffs
    diffusion_mechanism: str
    diffusion_category: str
    diffusion_coeffs: Coeffs
    id: str = ""

    def to(self, device: str) -> "MechanismTheta":
        return MechanismTheta(
            drift_mechanism=self.drift_mechanism,
            drift_category=self.drift_category,
            drift_coeffs={k: v.to(device) for k, v in self.drift_coeffs.items()},
            diffusion_mechanism=self.diffusion_mechanism,
            diffusion_category=self.diffusion_category,
            diffusion_coeffs={k: v.to(device) for k, v in self.diffusion_coeffs.items()},
            id=self.id,
        )


def mechanism_drift(x: torch.Tensor, theta: MechanismTheta) -> torch.Tensor:
    """True drift b(x) for a MechanismTheta. Same (..., d) -> (..., d) shape
    contract as sde_basis.parameterised_sde.drift_true(x, theta)."""
    mech = DRIFT_BY_NAME[theta.drift_mechanism]
    return mech.fn(x, theta.drift_coeffs)


def mechanism_diffusion(x: torch.Tensor, theta: MechanismTheta, eps: float = 1e-3) -> torch.Tensor:
    """True diagonal diffusion amplitude sigma(x) for a MechanismTheta. Same
    shape contract as sde_basis.parameterised_sde.sigma_diag_true(x, theta)."""
    mech = DIFFUSION_BY_NAME[theta.diffusion_mechanism]
    raw = mech.fn(x, theta.diffusion_coeffs)
    return raw.clamp_min(eps)


def sample_drift_instances(
    n_per_mechanism: int, d: int, generator: torch.Generator,
) -> List[Tuple[str, str, Coeffs]]:
    """Independently draw `n_per_mechanism` coefficient settings for every
    drift mechanism in the library."""
    instances = []
    for mech in DRIFT_MECHANISMS:
        for _ in range(n_per_mechanism):
            instances.append((mech.name, mech.category, mech.sampler(d, generator)))
    return instances


def sample_diffusion_instances(
    n_per_mechanism: int, d: int, generator: torch.Generator,
) -> List[Tuple[str, str, Coeffs]]:
    """Independently draw `n_per_mechanism` coefficient settings for every
    diffusion mechanism in the library."""
    instances = []
    for mech in DIFFUSION_MECHANISMS:
        for _ in range(n_per_mechanism):
            instances.append((mech.name, mech.category, mech.sampler(d, generator)))
    return instances


def generate_factorial_tasks(
    n_drift_per_mechanism: int,
    n_diffusion_per_mechanism: int,
    d: int,
    generator: torch.Generator,
    id_prefix: str = "fact",
) -> List[MechanismTheta]:
    """
    Full factorial cross of independently-sampled drift and diffusion
    instances: (#drift mechanisms x n_drift_per_mechanism) x (#diffusion
    mechanisms x n_diffusion_per_mechanism) total MechanismTheta tasks.
    """
    drift_instances = sample_drift_instances(n_drift_per_mechanism, d, generator)
    diffusion_instances = sample_diffusion_instances(n_diffusion_per_mechanism, d, generator)

    tasks: List[MechanismTheta] = []
    idx = 0
    for d_name, d_cat, d_coeffs in drift_instances:
        for s_name, s_cat, s_coeffs in diffusion_instances:
            tasks.append(MechanismTheta(
                drift_mechanism=d_name,
                drift_category=d_cat,
                drift_coeffs=d_coeffs,
                diffusion_mechanism=s_name,
                diffusion_category=s_cat,
                diffusion_coeffs=s_coeffs,
                id=f"{id_prefix}_{idx:05d}",
            ))
            idx += 1
    return tasks
