# Item 5: factorial drift/diffusion benchmark

New, additional data-generation functionality. Does **not** replace or
modify `data_gen/generate_meta_params.py` or the five item-1 shift regimes
(`none`, `drift_only`, `diffusion_only`, `combined`, `ood_1..ood_5`) --
those are untouched and items 1-4's already-queued real-GPU comparisons
still depend on them exactly as before.

## How the old scheme couples drift and diffusion

`data_gen/generate_meta_params.py` samples a `Theta(theta_b, theta_sigma)`
whose drift and diffusion coefficients ARE drawn independently
(`sample_theta` does two separate `torch.randn` calls) -- but both are
linear combinations of **one shared, fixed basis** for every single task in
every split and regime:

- drift: 8-term basis `[1, x, x^2, x^3, sin(x), cos(x), tanh(x), x^4]`
  (`sde_basis/basis_functions.py:drift_basis`)
- diffusion: 3-term basis `[1, |x|, sqrt(|x|)]` (`diffusion_basis`)

So the coupling isn't at the coefficient level, it's at the *functional
family* level: there's no way to express "an oscillatory drift paired with
multiplicative diffusion" as a qualitatively different mechanism from "a
double-well drift paired with constant diffusion" -- they're both just
different coefficients on the same basis. On top of that, the named shift
regimes (`cfg.theta_dist.shift_regimes`) each specify one joint
`(drift_shift, diffusion_shift, std_scale)` triple; even the
"drift_only"/"diffusion_only" regimes are two special-cased slices of one
shared 9-entry table, not independently-sampled severities.

## How the new library avoids that

`data_gen/mechanism_library.py` defines two independent libraries:

- **14 drift mechanisms** across the 5 required categories (2-3 each):
  - `stable_linear`: `ou_linear`, `piecewise_linear_asym`
  - `oscillatory` (periodic-drift potentials -- see below):
    `phase_oscillator`, `damped_periodic`, `dual_frequency_periodic`
  - `multistable` (finite polynomial-well potentials): `double_well`,
    `asym_double_well`, `steep_double_well`
  - `nonlinear_bounded` (saturating restoring force): `tanh_saturating`,
    `atan_saturating`, `rational_saturating`
  - `weakly_unstable`: `marginal_linear_unstable`, `weak_pitchfork`,
    `asym_weak_instability`
- **8 diffusion mechanisms** across the 3 required categories (2-3 each):
  - `approx_constant`: `const_diffusion`, `near_constant_weak_state`
  - `state_dependent`: `cir_sqrt`, `quadratic_state`, `abs_linear_state`
  - `multiplicative`: `gbm_multiplicative`, `gbm_with_floor`,
    `bounded_multiplicative`

Each mechanism has its own coefficient-distribution `sampler`. A task
(`MechanismTheta`) is built by drawing a drift instance (mechanism name +
sampled coefficients) and a diffusion instance **completely
independently**, then crossing them: 14 drift mechanisms x 4 coefficient
draws each = 56 drift instances; 8 diffusion mechanisms x 4 draws each = 32
diffusion instances; full factorial cross = **56 x 32 = 1792 tasks**.

Diagonal-drift note: like the existing `drift_true`, every mechanism here
depends only on its own state coordinate (no cross-dimension drift
coupling -- correlation is still handled by the Cholesky factor `L` on the
noise term). A true limit-cycle oscillator needs a coupled
position/velocity pair, which this diagonal-only convention can't express.
"Oscillatory" is therefore implemented as an overdamped periodic potential
(`b(x) = -grad` of a periodic function), the standard stand-in in this
setting -- genuinely periodic, infinite-well equilibria, as opposed to
`multistable`'s finite (2-3), polynomial-defined wells.

### Interface compatibility

`mechanism_drift(x, theta)` / `mechanism_diffusion(x, theta)` keep the
exact same shape contract as `drift_true(x, theta)` /
`sigma_diag_true(x, theta)` from `sde_basis/parameterised_sde.py`: both
take state `x` of shape `(..., d)` and a task object, return `(..., d)`.
`x_dim` (10), `T` (1.0) and `n_steps` (200) are read from
`config.base_config.cfg`, unchanged, so trajectories are structurally
identical to the existing pipeline's.

They do **not** reuse the fixed 8-term/3-term basis dot-product
implementation -- `MechanismTheta` stores a mechanism name + coefficients
specific to that mechanism's own functional form, dispatched by name
instead of a basis matrix. See `data_gen/mechanism_library.py`'s module
docstring for the full reasoning. One consequence: existing eval code that
hardcodes `from sde_basis.parameterised_sde import drift_true,
sigma_diag_true` (e.g. `adaptation/gated_finetuning_regularized.py`'s
`compute_mechanism_error`) does not automatically know how to score a
`MechanismTheta` -- a future consumer would need to call
`mechanism_drift`/`mechanism_diffusion` instead when the ground-truth task
is a `MechanismTheta` rather than a `Theta`. Wiring that up is out of scope
here (no NN/eval code changes in item 5).

Trajectories are simulated with `data_gen/simulate_true_sde.py`'s new
`simulate_batch_generic` -- the exact same Euler-Maruyama stepping,
correlated-noise, stability-check and defensive-freeze logic as the
existing `simulate_batch`, just parameterised by plain drift/diffusion
callables instead of a `Theta`. `simulate_batch` itself is untouched.

## Stability-skip guard

`data_gen/generate_factorial_benchmark.py` reuses the same
retry-until-count-or-`max_retries` loop as `data_gen/generate_trajectories.py`
(and the same `cfg.stability.max_state_abs` / `max_retries_per_theta`
settings), and extends it with a per-task outcome classification:

- **clean**: every role (`support`, `query`) hit its full requested count.
- **degraded**: some role got a partial but nonzero count.
- **unusable**: some role got zero usable trajectories -- this is the case
  the item-1 fix in `adaptation/gated_finetuning_regularized.py`
  (commit `cd61c66`) skips downstream rather than crashing on an empty
  stack.

`report_stability_stats` tallies clean/degraded/unusable by drift category,
by diffusion category, and by individual drift mechanism, so a degenerate
mechanism/category would show up clearly rather than being averaged away.

**Result on the full 1792-task benchmark**: all 1792/1792 tasks came back
**clean** -- every role (`support`, `query`) hit its full requested count
(10 + 32) on the first attempt, for every task. Zero `degraded`, zero
`unusable`. Verified directly against `data/factorial_index.csv` (every
`n_achieved == n_requested`) and cross-checked against the trajectory
files on disk (every `<task_id>/{support,query}/` directory has exactly
`n_achieved` `.npy` files, no missing/extra directories).

By drift category (clean / degraded / unusable / total):
```
  multistable           384 /    0 /    0 /  384
  nonlinear_bounded     384 /    0 /    0 /  384
  oscillatory            384 /    0 /    0 /  384
  stable_linear          256 /    0 /    0 /  256
  weakly_unstable        384 /    0 /    0 /  384
```

By diffusion category (clean / degraded / unusable / total):
```
  approx_constant        448 /    0 /    0 /  448
  multiplicative         672 /    0 /    0 /  672
  state_dependent        672 /    0 /    0 /  672
```

This confirms the 30-task moderate-scale verification's prediction
(stratified 2-per-category-pair, 30/30 clean including 6/6 for
`weakly_unstable` drift): at the existing pipeline's `T=1.0`,
`n_steps=200`, `max_state_abs=10.0` conventions, the weakly-unstable
coefficient ranges chosen here (e.g. `eps in [0.05, 0.4]` for
`marginal_linear_unstable`) are genuinely "weak" -- growth is
`~exp(eps * T) <= exp(0.4) ≈ 1.5x`, well inside the noise-driven excursion
range the guard is set up to catch. Unlike `ood_4`/`ood_5` in the old
scheme, no drift or diffusion category dominates a skip count here because
there is no skip count: the full-scale run produced zero unstable draws
across all 1792 tasks. If a future revision wants a nonzero skip rate to
stress-test the guard, widen `eps`/coefficient ranges in
`data_gen/mechanism_library.py`.

## Scale decision (item 5, step 6)

Moderate-scale check: 30 tasks (stratified across all 15
drift-category x diffusion-category pairs), full per-task trajectory count
(10 support + 32 query, matching `cfg.dataset_sizes.n_test_*_traj_per_theta`),
on CPU (`--device cpu`):

- **8.0s wall-clock total, 0.265s/task, 10.3 MB total (343 KB/task)**

Extrapolated to the full 1792-task factorial target:
**~7.9 minutes, ~615 MB**. Available disk at generation time: 3.5 GB free.
Both numbers are comfortably within reach on CPU alone -- no GPU is needed
for this step (unlike items 1-4's neural network training, which does need
GPU hardware and is separately still queued). **Decision: generate the
full-scale benchmark now, on CPU**, rather than deferring to GPU hardware.

## How to use this for real experiments

1. `data/factorial_meta_params.npz` holds `{"factorial": List[MechanismTheta]}`
   -- 1792 tasks, each with `drift_mechanism`/`drift_category`/
   `drift_coeffs`/`diffusion_mechanism`/`diffusion_category`/
   `diffusion_coeffs`/`id`.
2. `data/factorial_trajectories/<task_id>/{support,query}/traj_*.npy` holds
   the simulated trajectories, and `data/factorial_index.csv` indexes them
   (with `drift_category`/`diffusion_category`/`n_achieved`/`n_requested`/
   `attempts` columns, so downstream code can filter out `unusable` tasks
   the same way `gated_finetuning_regularized.py` already skips zero-count
   thetas for the existing regimes).
3. This pool is intended as the source for the **comprehensive main-
   experiment results** once items 1-4's components are selected --
   e.g. split it into train/val/test partitions, or (for item 6's
   compositional-generalisation experiment) hold out specific
   `(drift_mechanism, diffusion_mechanism)` cells. Neither split is built
   here -- item 5 only builds and verifies the underlying pool.
4. The existing five-regime setup (`none`/`drift_only`/`diffusion_only`/
   `combined`/`ood_1..5`) remains the right tool for the **fast
   component-selection comparisons** already queued for items 2-4 (loss
   variant, learned gate, meta-learned adaptation rates) -- it's cheap (9
   regimes x 30 test thetas) and already wired into the existing eval code
   path (`drift_true`/`sigma_diag_true`, `compute_mechanism_error`, etc.).
   Once a component is selected using that fast loop, re-run the winning
   configuration against this factorial benchmark for the paper's headline
   numbers -- but that requires a `MechanismTheta`-aware eval path (see
   "Interface compatibility" above), which is future work, not part of
   item 5.
5. To regenerate at a different scale: `python -m
   data_gen.generate_factorial_benchmark --stage params
   --n-drift-per-mechanism K1 --n-diffusion-per-mechanism K2` (total tasks
   = 14*K1 x 8*K2), then `--stage full` to simulate all of them, or
   `--stage verify` first to re-check timing/disk on new settings before
   committing to a larger `full` run.
