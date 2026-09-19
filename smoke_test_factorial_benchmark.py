# smoke_test_factorial_benchmark.py
#
# Item 5, step 6 precursor: smallest-possible smoke test for the factorial
# drift/diffusion mechanism library (data_gen/mechanism_library.py) and its
# generator (data_gen/generate_factorial_benchmark.py).
#
# Confirms, at toy scale:
#   - every drift/diffusion mechanism's sampler + fn run end-to-end and
#     produce finite, correctly-shaped output for every mechanism in the
#     library (not just whichever one a random factorial draw happens to
#     hit)
#   - diffusion mechanisms are strictly positive
#   - the factorial cross-product produces the expected count
#   - simulate_batch_generic (the new EM integrator sibling used instead of
#     touching simulate_batch) actually produces trajectories and a
#     stability mask of the right shape
#   - the stratified-subset + stability-stats reporting machinery runs
#     without crashing on a tiny synthetic run
#
# This is NOT the moderate-scale timing/disk verification required by item 5
# step 6 -- see FACTORIAL_BENCHMARK.md for that (real, non-toy) run and its
# results.
#
# Usage: python smoke_test_factorial_benchmark.py

import math
import shutil
import tempfile

import torch

from data_gen.mechanism_library import (
    DRIFT_MECHANISMS,
    DIFFUSION_MECHANISMS,
    generate_factorial_tasks,
    mechanism_drift,
    mechanism_diffusion,
)
from data_gen.simulate_true_sde import simulate_batch_generic
from sde_basis.covariance import sample_correlation


def check_every_mechanism_runs(d: int = 3, batch: int = 5):
    gen = torch.Generator()
    gen.manual_seed(0)
    x = torch.randn(batch, d, generator=gen) * 2.0  # exercise x<0, x>0, |x|>1

    for mech in DRIFT_MECHANISMS:
        coeffs = mech.sampler(d, gen)
        for name, v in coeffs.items():
            assert v.shape == (d,), f"{mech.name}.{name} has shape {v.shape}, expected ({d},)"
        b = mech.fn(x, coeffs)
        assert b.shape == x.shape, f"{mech.name} output shape {b.shape} != {x.shape}"
        assert torch.isfinite(b).all(), f"{mech.name} produced non-finite drift: {b}"

    for mech in DIFFUSION_MECHANISMS:
        coeffs = mech.sampler(d, gen)
        for name, v in coeffs.items():
            assert v.shape == (d,), f"{mech.name}.{name} has shape {v.shape}, expected ({d},)"
        s = mech.fn(x, coeffs)
        assert s.shape == x.shape, f"{mech.name} output shape {s.shape} != {x.shape}"
        assert torch.isfinite(s).all(), f"{mech.name} produced non-finite diffusion: {s}"
        assert (s >= 0).all(), f"{mech.name} produced negative diffusion before flooring: {s}"

    print(f"All {len(DRIFT_MECHANISMS)} drift and {len(DIFFUSION_MECHANISMS)} diffusion "
          f"mechanisms ran cleanly (finite, correctly-shaped output).")


def check_factorial_cross_and_simulation():
    d = 3
    gen = torch.Generator()
    gen.manual_seed(1)

    n_drift_per_mechanism = 2
    n_diffusion_per_mechanism = 2
    tasks = generate_factorial_tasks(n_drift_per_mechanism, n_diffusion_per_mechanism, d, gen)
    expected = (len(DRIFT_MECHANISMS) * n_drift_per_mechanism) * (
        len(DIFFUSION_MECHANISMS) * n_diffusion_per_mechanism
    )
    assert len(tasks) == expected, f"expected {expected} factorial tasks, got {len(tasks)}"
    assert len(set(t.id for t in tasks)) == len(tasks), "task ids are not unique"

    theta = tasks[0]
    cov_gen = torch.Generator()
    cov_gen.manual_seed(42)
    _, L = sample_correlation(d, cov_gen, "cpu")

    x0 = torch.randn(6, d, generator=gen) * 0.25
    sim_gen = torch.Generator()
    sim_gen.manual_seed(2)

    drift_fn = lambda x: mechanism_drift(x, theta)
    diffusion_fn = lambda x: mechanism_diffusion(x, theta)

    trajs, mask = simulate_batch_generic(
        drift_fn, diffusion_fn, L, x0, T=0.1, n_steps=10, x_max_abs=10.0, generator=sim_gen,
    )
    assert trajs.shape == (6, 11, d), f"unexpected trajectory shape {trajs.shape}"
    assert mask.shape == (6,)
    assert mask.dtype == torch.bool

    print(f"Factorial cross produced {len(tasks)} tasks (expected {expected}); "
          f"simulate_batch_generic produced trajectories of shape {tuple(trajs.shape)}, "
          f"{mask.sum().item()}/{mask.shape[0]} valid.")


def check_generator_stratified_and_stats():
    from data_gen import generate_factorial_benchmark as gfb

    tmp_root = tempfile.mkdtemp(prefix="smoke_test_factorial_")
    try:
        gfb.cfg.basis.x_dim = 2
        gfb.cfg.time_grid.n_steps = 6
        gfb.cfg.stability.max_retries_per_theta = 3
        gfb.cfg.dataset_sizes.n_test_support_traj_per_theta = 2
        gfb.cfg.dataset_sizes.n_test_query_traj_per_theta = 2

        tasks = gfb.build_tasks(n_drift_per_mechanism=1, n_diffusion_per_mechanism=1, seed=7)
        subset = gfb.stratified_subset(tasks, per_pair=1)
        n_pairs = len({(t.drift_category, t.diffusion_category) for t in tasks})
        assert len(subset) == n_pairs, f"expected {n_pairs} stratified tasks, got {len(subset)}"

        roles = gfb._roles()
        index_rows, elapsed, total_bytes = gfb.simulate_tasks(subset, tmp_root, roles, device="cpu")
        assert elapsed >= 0.0
        assert len(index_rows) > 0, "no index rows produced"
        assert all(math.isfinite(r["n_achieved"]) for r in index_rows)

        gfb.report_stability_stats(subset, index_rows)
        print(f"\nGenerator smoke check: {len(subset)} stratified tasks simulated, "
              f"{len(index_rows)} index rows, {total_bytes} bytes written.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    check_every_mechanism_runs()
    check_factorial_cross_and_simulation()
    check_generator_stratified_and_stats()
    print("\nFactorial benchmark smoke test passed. This confirms the mechanism "
          "library, generic EM integrator, and generator wiring are correct at "
          "toy scale -- it does NOT confirm timing/disk feasibility at real "
          "scale; see FACTORIAL_BENCHMARK.md for that.")


if __name__ == "__main__":
    main()
