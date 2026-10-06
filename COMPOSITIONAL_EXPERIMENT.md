# Item 6: compositional generalisation as a headline experiment

This document covers the design, the exact commands for the real run, and
what this change does and does not establish. It does not report any
experimental result -- no real-scale run has been executed (see "What is
NOT established" below).

## Phase 0 finding: the data mismatch

`data/factorial_index.csv` (item 5's factorial benchmark) has no `split`
column and no `file_path` column, and only `support`/`query` roles on
disk. The legacy pipeline (`dataloaders/trajectory_datasets.py`'s
`TrajectoryDataset`, `training/train_meta.py`'s `train_meta_loop`) expects
a `split` column and `train_inner`/`val_inner` roles (many more
trajectories per task, used to train the encoder/sde/head themselves, not
for few-shot test-time adaptation). Phase 1 below adds the missing
plumbing without touching either existing pipeline.

## Design

### Split rule (phase 2, `data_gen/compositional_splits.py`)

- Unit of holdout: a `(drift_mechanism, diffusion_mechanism)` **pair** --
  14 drift mechanisms x 8 diffusion mechanisms = 112 pairs, each containing
  16 tasks (4 independent drift coefficient draws x 4 independent
  diffusion coefficient draws, item 5's default generation settings).
- Holdout fractions 0.10 / 0.25 / 0.50 give exactly 11 / 28 / 56 held-out
  pairs (`round(fraction * 112)`), i.e. 176 / 448 / 896 held-out test
  tasks -- all comfortably over the 100-task minimum (item 8).
- **Coverage constraint**: a pair is only eligible to be held out if doing
  so leaves >=1 remaining training pair in both its row (drift mechanism)
  and its column (diffusion mechanism) -- so every mechanism is seen in at
  least one training pair, and the heldout-test score measures
  recombination of known mechanisms, not generalisation to an unseen one.
  Held-out pairs are selected greedily, preferring the row/column currently
  furthest below its even per-row/per-column holdout share, so holdouts
  don't cluster onto a handful of mechanisms.
- **ID-test**: for every training pair, 2 of its 16 tasks (deterministic
  given the seed) are set aside as `id_test` -- never trained on, but from
  a pair the model DID see. This is what `experiments/aggregate_compositional.py`'s
  ID-vs-heldout gap compares against.
- Determinism: `build_split(fraction, seed)` is a pure function of
  `(fraction, seed)` plus the on-disk task list; same seed -> byte-identical
  split, different seed -> (checked, not just assumed) a different split.
- Committed manifests (seed 0, the default split used by every training
  seed -- see "seed semantics" below) are under
  `config/compositional_splits/split_frac{10,25,50}_seed0.json`.

### Arm mapping (phase 3, `adaptation/compositional_arms.py`)

| # | Professor's arm | Code | Encoder | Adaptation |
|---|---|---|---|---|
| 1 | original single-z Meta-SDE | `single_z` | `models/single_latent_encoder.py`'s `SingleLatentEncoder` (ONE latent, default dim 16, 32 selectable as item 23's matched-capacity control) | fixed 50-step Adam, **one leaf tensor** passed as both the drift and diffusion argument (so `||Delta z_f|| == ||Delta z_g||` exactly, by construction, not by coincidence) + the original scalar gate |
| 2 | factorised z_f, z_g | `factorised` | `models/encoder.py`'s `TrajEncoder` (unchanged) | `adaptation/gated_finetuning_regularized.adapt_model` (unchanged, reused verbatim) + the existing scalar gate |
| 3 | factorised + learned adaptation | `factorised_learned_rates` | `TrajEncoder` | item 4's differentiable K-step adaptation, `adaptation_mode="conditioned"` (task-conditioned rates, no gate -- `a_f`/`a_g` logged as NaN) |
| 4 | factorised + learned selective adaptation | `factorised_selective` | `TrajEncoder` | item 4's differentiable K-step adaptation, `adaptation_mode="conditioned_plus_gate"` (task-conditioned rates + a fresh `(a_f, a_g)` controller) |

This is the literal mapping in the professor's task list (item 6's four
bullets correspond 1:1 to item 4's four `--adaptation-mode` variants plus
the pre-item-1 architecture). An alternative considered and rejected: use
item 3's *detached* `LearnedGateController` (trained against the 50-step
Adam path) for arm 4 instead of item 4's differentiable K-step
`conditioned_plus_gate`. Rejected because item 3's controller was
explicitly trained against a different, detached inner loop (see
`training/train_controller.py`'s "Design choice" note) and mixing it with
a from-scratch base checkpoint here would not be a meaningful comparison;
item 4's `conditioned_plus_gate` already *is* "learned selective
adaptation" end to end.

### Why new files instead of editing existing ones

Every new module in this change (`dataloaders/factorial_dataset.py`,
`evaluation/compositional_metrics.py`,
`data_gen/generate_factorial_training_trajectories.py`,
`adaptation/compositional_arms.py`, `training/train_compositional.py`) is
additive. None of `training/train_meta.py`, `training/train_controller.py`,
`training/train_diff_adaptation.py`,
`adaptation/gated_finetuning_regularized.py`,
`adaptation/differentiable_adaptation.py`, or `models/encoder.py` were
modified -- items 1-4's already-queued real-GPU comparisons depend on them
behaving exactly as before, and this change's own existing-smoke-test
re-runs (see below) confirm they still do.

Two building blocks ARE reused directly, unmodified, because they are
already generic over latent dimension and never touch the ground-truth
theta type: `adapt_model` and `compute_residual`
(`adaptation/gated_finetuning_regularized.py`), plus `k_step_adapt`,
`get_rates`, and `initial_grad_norms`
(`adaptation/differentiable_adaptation.py`).

One piece could NOT be reused directly: `compute_mechanism_error`
(legacy) and `differentiable_adapt`'s internal call to it hardcode the
basis-coefficient `Theta` ground truth (`drift_true`/`sigma_diag_true`),
which doesn't know how to score a `MechanismTheta`. Rather than branch
that existing function, `evaluation/compositional_metrics.py` and
`adaptation/compositional_arms.py::differentiable_adapt_factorial`
duplicate the small amount of logic that needs the swap (same pattern
`data_gen/simulate_true_sde.py`'s `simulate_batch_generic` already uses
for the same reason).

### `training/train_compositional.py` vs. extending the legacy training scripts

Phase 1d asks that "training and adaptation entry points accept an
explicit list of training/test task ids." The legacy entry points
(`train_meta_loop`, `controller_train_loop`, `diff_adaptation_train_loop`,
`gated_finetuning_regularized.main`) are tightly coupled to regime-based
data loading (`TrajectoryDataset` by split name, `meta_params["test<regime>"]`)
and are relied on unchanged by items 1-4. Rather than thread an
arm/task-id-list branch through them, `training/train_compositional.py`
and `experiments/run_compositional.py` are new entry points whose
*primary* interface is an explicit task-id list (`FactorialTrajectoryDataset(task_ids=...)`,
`train_adapt(..., train_task_ids=...)`, the compositional split's
`task_ids["train"]`/`["id_test"]`/`["heldout_test"]`) -- satisfying the
intent of phase 1d without risking the legacy scripts' behaviour.

### Seed semantics

`seed` in `experiments/run_compositional.py` is a **training** seed (model
init, optimizer/adaptation RNG) -- the compositional **split** (which
pairs are held out) is controlled separately by `--split-seed` (default
0, the committed manifest), so every training seed for a given fraction is
evaluated against the identical train/id_test/heldout_test partition. This
isolates training-seed variance (item 8's "5-10 independent training
seeds") from split variance, rather than conflating the two.

### Train-base data source (phase 1c)

`data/factorial_trajectories/` (item 5) only has `support`(10)/`query`(32)
per task -- enough for `train_adapt`/`evaluate` (which use few-shot
adaptation), but not for `train_base` (which needs many `train_inner`
trajectories per task, the same role `training/train_meta.py` consumes).
`data_gen/generate_factorial_training_trajectories.py` adds exactly that,
to a NEW directory (`data/factorial_train_trajectories/`) and a NEW index
(`data/factorial_train_index.csv`), reusing item 5's `simulate_tasks` as-is.

**This has not been run at full scale.** A tiny 5-task timing test (real
settings: `x_dim=10`, `n_steps=200`, 80 `train_inner` + 20 `val_inner`
trajectories/task) gave 0.164 s/task, 817 KB/task, extrapolating to
**~4.9 minutes and ~1.46 GB for all 1792 tasks** -- comfortably under the
~2.9 GB free at generation time, but still not run for real (the smallest
training set, at 10% holdout, needs ~1414 of the 1792 tasks, so somewhat
less than this upper bound).

## Exact commands for the real run, in order

```bash
# 1. (Already done, item 5) Full factorial benchmark + support/query trajectories:
#    data/factorial_meta_params.npz, data/factorial_index.csv, data/factorial_trajectories/

# 2. Generate train_inner/val_inner trajectories for ALL 1792 tasks (NOT run here --
#    review the verify stage's time/disk estimate on your hardware first):
python -m data_gen.generate_factorial_training_trajectories --stage verify \
    --task-ids-file <(python -c "import pandas as pd; print('\n'.join(sorted(pd.read_csv('data/factorial_index.csv')['theta_id'].unique())))")
python -m data_gen.generate_factorial_training_trajectories --stage full --confirm-full-run \
    --task-ids-file <same file as above>

# 3. (Already done) Compositional splits for fractions 0.10/0.25/0.50, seed 0:
python -m data_gen.compositional_splits --seed 0
#    (committed under config/compositional_splits/)

# 4. Inspect the full run matrix before committing GPU time:
python -m experiments.run_compositional --dry-run
#    -> 3 fractions x 4 methods x 5 seeds = 60 (method, fraction, seed) runs
#       (30 of which also need a train_adapt stage)

# 5. For every (method, fraction, seed) in the matrix:
python -m experiments.run_compositional --stage train_base  --method <m> --fraction <f> --seed <s>
python -m experiments.run_compositional --stage train_adapt --method <m> --fraction <f> --seed <s>  # no-op for single_z/factorised
python -m experiments.run_compositional --stage evaluate    --method <m> --fraction <f> --seed <s>
#    (results accumulate into results/compositional_results.csv)

# 6. Aggregate:
python -m experiments.aggregate_compositional --results-csv results/compositional_results.csv
```

CPU smoke test (no real data needed, builds its own tiny environment):
```bash
python -m experiments.run_compositional --toy --stage train_base  --methods factorised --fractions 0.5 --seeds 0
python -m experiments.run_compositional --toy --stage train_adapt --methods factorised --fractions 0.5 --seeds 0
python -m experiments.run_compositional --toy --stage evaluate    --methods factorised --fractions 0.5 --seeds 0
python smoke_test_compositional.py   # full, in-process version of the above, all 4 arms
```

## Estimated run count and compute

- `--dry-run` reports **60 (method, fraction, seed) runs** for the default
  3 fractions x 4 methods x 5 seeds (30 of which also need a `train_adapt`
  stage). Each run's `train_base` is one full meta-training pass over the
  fraction's training tasks (comparable in cost to
  `training/train_meta.py`'s existing 50-epoch run, just over a
  differently-sized task pool per fraction/method); `train_adapt` (for 2 of
  the 4 methods) is comparable to `training/train_diff_adaptation.py`'s
  existing per-mode meta-training cost; `evaluate` is comparable to
  `adaptation/gated_finetuning_regularized.py`'s existing per-task
  adaptation+eval cost, run over (`id_test` + `heldout_test`) tasks (176 to
  896+ tasks depending on fraction). No part of this has been run at real
  scale -- see "What is NOT established" below.

## What is NOT established

- **No experimental result.** No real-scale `train_base`/`train_adapt`/
  `evaluate` run has been executed -- only toy-scale smoke tests (x_dim=2,
  a couple of mechanism draws, 1-2 epochs/meta-steps). Nothing here says
  single_z, factorised, factorised_learned_rates, or factorised_selective
  is better than any other at real scale, on held-out pairs or otherwise.
- **The train_inner/val_inner trajectories for train_base don't exist yet
  at real scale** (see "Train-base data source" above) -- `train_base`
  cannot currently be run for real until
  `generate_factorial_training_trajectories.py --stage full` is run.
- **Which loss variant to use for train_base is not re-decided here.**
  `training/train_compositional.py::train_base` defaults to
  `trajectory_mse` (item 2's unchanged default) and exposes `loss_variant`
  as a plain setting, per the task description -- item 2's real GPU
  comparison (if it changes the project-wide default) should be re-applied
  here, not decided by this change.
- **z_dim=32 single_z (item 23's matched-capacity control) is wired up
  (`--z-dim 32 --method single_z`) but not run or compared.**
- The paired-comparison / bootstrap-CI machinery in
  `experiments/aggregate_compositional.py` is a plain percentile bootstrap
  over task-level rows plus a seed-level standard error -- appropriate for
  item 8's stated bar ("confidence intervals or standard errors across
  tasks/seeds"), not a claim of a more sophisticated statistical test.
