# Agent instructions — Yu Cao's package (Positive control)

You are working inside the Meta-Diffusion repository on branch `yu-cao`, on a Mac with no
CUDA device. Read `handoff/Yu-Cao/README.md` for the scientific brief and
`docs/PROTOCOL_CARD.md` for the values you may not change. This file states the rules that
must not be broken.

## Environment

- macOS. Run `bash handoff/Yu-Cao/setup.sh` once before anything else.
- Interpreter: `.venv/bin/python`. The shell drivers pick it up automatically.
- **CPU only.** `runner/stage1_gmm.py` already falls back to CPU when CUDA is absent, so
  nothing needs changing. Do not add an MPS device path: the stage-1 backbone is a
  four-layer perceptron on two-dimensional points, where Metal dispatch overhead usually
  makes it slower than the CPU, and a device change would make your numbers
  non-comparable with everyone else's for no gain.
- No dataset is required. Stage 1 is synthetic. Do not set `CIFAR100_ROOT` and do not
  download CIFAR-100.
- Run everything from the repository root.

## Files you may create or modify

```
scripts/stage1_panel.py            (new)
scripts/plot_panel.py              (new)
scripts/stage1_missing_metrics.py
scripts/run_relatedness_sweep.sh
runner/stage1_gmm.py
handoff/Yu-Cao/results/*
artifacts/*.log                    (the sweep script already writes here; these are tracked)
```

## Files you must not touch, for any reason

```
models/    training/    diagnostics/    config/    episodes/    adaptation/    domains/
evaluation/instruments.py    scripts/cifar_table.py    scripts/phase_d_sweeps.sh
artifacts/*.json             handoff/Jinjian/    handoff/Jing-Peng/
```

Another person owns each. If you find a bug in one, write it down in your results file and
carry on. Do not fix it, and do not copy the file to work around it.

## Hard rules for the measurement

1. **All four coordinates share one noising draw per task.** Use
   `ScoreModel.eps_hat_many`, which takes a list of coordinates against a single
   `(x_t, t, eps)`. Separate `q_sample` calls throw away a factor of sixty-five in
   variance and make the effect unmeasurable.
2. **The substitution coordinate is a mean over tasks, not over the batch.** Every query
   point within one task already shares one coordinate, so shuffling or averaging along
   the batch dimension is a no-op. Form `z̄` across the evaluation tasks.
3. **Report `delta_task` as an absolute paired difference with a 95 % interval**, plus the
   step count. Never headline `task_specific_frac`: its denominator decays to zero and
   changes sign.
4. **Rebuild from the checkpoint's own stored config**, never from current defaults.
5. **A consistent step count across the five relatedness points matters more than a large
   one.** If you shorten the runs, shorten all five and say so.

## Rules for `scripts/plot_panel.py`

It is used by two other people on logs you will never see, so:

- read `delta_task`, `r_basis`, `z_spread_rel`, `z_center_norm` by those exact names;
- **tolerate every one of them being absent** — none of the eighteen logs currently on
  disk has `delta_task`, because the control was added after those runs finished. Draw
  what exists, annotate what does not, never crash and never plot a silent zero;
- put `delta_task` on top and make it visually dominant; raw losses, if drawn at all, are
  subordinate and labelled as such;
- shade the final fifth of the run, which is the only window a number may be read from.

## Order of work

1. `bash handoff/Yu-Cao/setup.sh` then `bash handoff/Yu-Cao/verify.sh`.
2. Extend the stage-1 metrics with the mean-coordinate control and the profile.
3. Train one 80 000-step stage-1 checkpoint (about 24 minutes) and run the panel on it.
4. Start the relatedness sweep — roughly two hours; leave it running.
5. Run the panel on all five sweep checkpoints; plot `delta_task` against `perturb`.
6. Write `scripts/plot_panel.py` while the sweep runs.
7. Copy results into `handoff/Yu-Cao/results/`.

`bash handoff/Yu-Cao/run_all.sh` runs steps 3 to 5.

## What you must not do

- Do not add a device path, a dependency, or a network. matplotlib and torch are what you
  have.
- Do not decide whether the method works. Deliver numbers; the decision happens at
  integration against criteria fixed in advance.
- Do not commit `*.txt` or `*.png` under `artifacts/` — they are ignored there and will
  silently fail to appear. `artifacts/*.log` is the exception and is tracked.
- Do not push to a branch that is not `yu-cao`.
