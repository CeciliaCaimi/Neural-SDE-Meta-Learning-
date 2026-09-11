# Yu Cao — Positive control

**Machine:** Mac, no CUDA. Everything here runs on the CPU.
**Budget:** about 3 hours of development, 3 hours of CPU that you can leave running.
**Branch:** `yu-cao`

Your package needs no GPU, no dataset, and no file from anyone else's machine. Stage 1 is
two-dimensional Gaussian mixtures — synthetic data, an analytic ground truth, and a
four-layer perceptron instead of a U-Net. Measured on the reference machine's CPU it runs
at **54 iterations per second**, so 80 000 steps take 24 minutes with no GPU at all.

It is also the scientifically decisive part of this round, which is worth understanding
before you start.

## Why this matters

Every measurement on natural images says the task coordinate contributes nothing
task-specific. But nobody has ever measured the same quantity in the arm where the
mechanism **does** work. Stage 1 computes the correct coordinate against zero and against
a shuffled coordinate — `scripts/stage1_missing_metrics.py` does exactly that — and it
has never computed the **mean-coordinate substitution**, which is the control that
settles the question.

So a zero on CIFAR currently has no calibration. Nobody knows whether the diagnostic can
produce a large positive number at all, in this codebase, on any domain. You are going to
find out. If it is large and positive on stage 1 and zero on CIFAR, the CIFAR result is
about the task family. If it is *also* near zero on stage 1 — where transport
demonstrably beats every baseline — then the diagnostic itself is suspect and much of the
last month's reasoning needs revisiting.

Then you dial the one variable images cannot dial: how far apart the tasks are.

---

## Task 1 — the four-way panel on stage 1

`scripts/stage1_missing_metrics.py` already reports, for a stage-1 checkpoint: `r_basis`,
the loss under the correct coordinate, under `z = 0`, under a shuffled coordinate, and
the coordinate spread. Extend it, or write `scripts/stage1_panel.py` alongside it, to add:

- **the mean-coordinate substitution**: one coordinate shared by every task, formed as the
  mean over the evaluation tasks, and `delta_task = L(z̄) − L(z_own)`;
- **per-task paired differences with 95 % intervals**, not just means — all four
  coordinates must share one noising draw per task, which `ScoreModel.eps_hat_many` gives
  you in a single call;
- **the coordinate-loss profile**: `L(z̄ + s·(z_own − z̄))` and `L(z̄ + s·(z_j − z̄))` for
  `s` in {0, ½, 1, 1½, 2} and some other task `j`. On CIFAR the expectation is flat along
  the correct ray; here we expect a clear minimum at `s = 1`. The contrast between the two
  is the point.

Run it on a fresh 80 000-step checkpoint, and report the step count with the number.

## Task 2 — how the effect scales with task distance

The stage-1 task family has a dial that images do not: `--family related --perturb p`
generates tasks that are perturbations of a common family, with `p = 0` meaning every task
is identical and `p = 1` meaning mutually unrelated. `scripts/run_relatedness_sweep.sh`
already trains the five settings; it costs about two hours of CPU and you can leave it.

Then apply Task 1's panel to each of the five checkpoints and produce the plot this
project does not yet have: **`delta_task` against task distance**, with intervals.

The repository states a falsifiable claim: *the method needs a task family whose members
lie farther apart than the shared backbone can absorb.* That claim predicts `delta_task`
rises monotonically with `p`. Nobody has tested it. Your five points are the test, and
they cost two hours on a laptop rather than a week on a cluster.

If you are short of time, run 40 000 steps instead of 80 000 for the sweep and say so — a
consistent step count across the five points matters far more than its absolute value.

## Task 3 — the shared plotting tool

Write `scripts/plot_panel.py`, which everyone uses at integration. It reads a training log
(`checkpoints/<run>_log.jsonl`, one JSON object per line, diagnostic rows carrying a
`diagnostics` key) and draws two stacked panels:

- **top**: `delta_task` against step, with a 95 % interval band. This is the headline and
  must be visually dominant.
- **bottom**: `r_basis`, `z_spread_rel` and `z_center_norm` against step.

Three requirements that are not cosmetic:

1. **Never plot a raw loss curve as the headline.** Over steps 31k–50k of one run the
   standard deviation of the raw loss between diagnostic points is 0.0241, while the
   paired difference has a standard deviation of 0.00037 — sixty-five times smaller. A raw
   loss curve cannot show this effect and will mislead anyone who reads it.
2. **Tolerate missing fields.** `delta_task` and `z_center_norm` appear only in runs from
   another package, and none of the eighteen logs already on disk has them. Draw what is
   present and say what is absent; do not crash and do not silently plot zeros.
3. **Mark the convergence window.** Shade the final fifth of training, which is the only
   region any number may be read from.

matplotlib is already in `requirements.txt` and is used by the figure scripts only.

---

## Your files

Yours to modify: `scripts/stage1_missing_metrics.py`, `scripts/run_relatedness_sweep.sh`,
`runner/stage1_gmm.py`.
Yours to create: `scripts/stage1_panel.py`, `scripts/plot_panel.py`.

**Do not touch** `models/`, `training/`, `diagnostics/`, `config/`, `episodes/`,
`adaptation/`, `scripts/cifar_table.py`, `scripts/phase_d_sweeps.sh`, or
`evaluation/instruments.py`. Two other people own those.

Note that `diagnostics/controls.py` is not yours — but you also do not need it. Stage 1
has always carried its own metrics. Your implementation of the mean-coordinate control
will be the third written independently in this project, and agreement between the three
is the only cheap check available on a quantity that has already been misread twice.

## How to know you are finished

- `delta_task` on a converged stage-1 checkpoint has a mean, a 95 % interval and a step
  count, and you can say plainly whether it is positive.
- The coordinate-loss profile is plotted for both rays.
- Five relatedness points exist at a consistent step count, each with an interval.
- `scripts/plot_panel.py` runs on a log that has none of the new fields — try it on
  `checkpoints/*_log.jsonl` if you have one, or on a two-line file you write by hand — and
  produces a figure with a note rather than a traceback.

## Deliver into `handoff/Yu-Cao/results/`

`stage1_panel.txt`, `stage1_relatedness.txt`, `stage1_profile.png`,
`delta_task_vs_distance.png`, and `plot_panel.py`'s output on one example log.

Not into `artifacts/` — it ignores `*.txt` and `*.png`, and your files would silently fail
to commit. The stage-1 console logs are the exception: `artifacts/*.log` is tracked, and
the sweep script writes there already.

## Finishing

```bash
git add scripts runner artifacts/*.log handoff/Yu-Cao/results
git commit -m "Stage-1 positive control and the task-distance sweep"
git push origin yu-cao
```

Push to `yu-cao` and nowhere else. Do not merge into `main` and do not rebase onto another
package's branch. Stage-1 checkpoints are small, but they still live under
`checkpoints/`, which is ignored — the console logs in `artifacts/` are the evidence that
travels.

**Do not write a conclusion.** Deliver the numbers.
