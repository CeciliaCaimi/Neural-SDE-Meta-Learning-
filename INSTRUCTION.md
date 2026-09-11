# Instruction — Yu Cao

Follow this in order. Your package is the **positive control**: the same measurement
everyone else is making on natural images, made in the one domain where the mechanism
demonstrably works. No GPU, no dataset, no file from anyone else's machine.

The reasoning behind each step is in `handoff/Yu-Cao/README.md`. The values you may not
change are in `docs/PROTOCOL_CARD.md`. This file is the sequence.

---

## Why this matters, in three sentences

Every measurement on CIFAR-100 says the task coordinate contributes nothing
task-specific. But that quantity has never been measured where it is known to be
positive: stage 1 computes the correct coordinate against zero and against a shuffled
coordinate, and has never computed the mean-coordinate substitution. So the zero on
images has no calibration, and nobody yet knows whether this diagnostic can produce a
large positive number at all, in this codebase, on any domain.

## Step 1 — Clone and bootstrap

```bash
git clone -b yu-cao https://github.com/CeciliaCaimi/Neural-SDE-Meta-Learning-.git
cd Neural-SDE-Meta-Learning-
bash handoff/Yu-Cao/setup.sh
bash handoff/Yu-Cao/verify.sh
```

The preflight ends by measuring this Mac's throughput and telling you how long a run will
take here. Use that number to choose your step count in step 4. Do not continue past a
`FAIL`.

Two things you should **not** do: do not download CIFAR-100 or set `CIFAR100_ROOT` — stage
1 generates its Gaussian mixtures from a seed. And do not add an MPS device path: the
backbone is a four-layer perceptron on two-dimensional points, where Metal dispatch
overhead usually loses to the CPU, and a device change would make your numbers
non-comparable with everyone else's for no gain.

## Step 2 — Add the mean-coordinate control (about 1.5 hours)

`scripts/stage1_missing_metrics.py` already reports basis usage, the correct coordinate
against zero, a shuffled coordinate, and coordinate spread. Extend it, or write
`scripts/stage1_panel.py` beside it, to add:

- **the mean-coordinate substitution** — one coordinate shared by every task, and
  `delta_task = L(z̄) − L(z_own)`;
- **per-task paired differences with 95 % intervals**, all four coordinates sharing one
  noising draw per task, which `ScoreModel.eps_hat_many` gives you in a single call;
- **the coordinate-loss profile**: `L(z̄ + s·(z_own − z̄))` and `L(z̄ + s·(z_j − z̄))` for
  `s ∈ {0, ½, 1, 1½, 2}` and some other task `j`.

**Take the mean across tasks, not across the batch.** Every query point within one task
already shares one coordinate, so averaging or shuffling along the batch dimension is a
no-op. This is the single easiest way to get a wrong answer here.

On images the profile is expected to be flat along the correct ray. Here a clear minimum
at `s = 1` is expected. The contrast between the two is the point of the whole exercise.

## Step 3 — The base run (about 25 minutes)

```bash
bash handoff/Yu-Cao/run_all.sh
```

It trains one stage-1 checkpoint and runs your panel on it. Look at whether `delta_task`
is clearly positive, and report it with its interval and the step count.

## Step 4 — The task-distance sweep (leave it running)

The stage-1 family has a dial images do not: `--family related --perturb p` interpolates
from every task identical to mutually unrelated.

```bash
STAGE1_STEPS=40000 bash handoff/Yu-Cao/run_all.sh sweep
```

Choose the step count from what the preflight measured — but **all five points must use
the same one**, and if you change it, say so in the results file. A consistent step count
matters far more than a large one.

Then plot `delta_task` against `perturb`, with intervals. The repository states a
falsifiable claim — *the method needs a task family whose members lie farther apart than
the shared backbone can absorb* — which predicts that curve rises monotonically. Nobody
has tested it. Five points on a laptop are the test.

## Step 5 — The shared plotting tool (write it while the sweep runs)

`scripts/plot_panel.py`, used by everyone at integration. Two stacked panels: `delta_task`
against step with a 95 % interval band on top and visually dominant; below it `r_basis`,
`z_spread_rel` and `z_center_norm`.

Three requirements that are not cosmetic:

1. **Never plot a raw loss curve as the headline.** Over one run's converged stretch the
   standard deviation of the raw loss between diagnostic points is 0.0241, while the
   paired difference has a standard deviation of 0.00037 — sixty-five times smaller. A raw
   loss curve cannot show this effect and will mislead whoever reads it.
2. **Tolerate every new field being absent.** `delta_task` and `z_center_norm` appear only
   in runs from another package, and none of the eighteen logs already on disk has either.
   Draw what is present, annotate what is not, never crash and never plot a silent zero.
3. **Shade the final fifth of training** — the only window a number may be read from.

Read the fields by those exact names. Another package emits them and the two of you are
not talking.

## Step 6 — Hand back

```bash
git add scripts runner artifacts/*.log handoff/Yu-Cao/results
git commit -m "Stage-1 positive control and the task-distance sweep"
git push origin yu-cao
```

Deliverables go in `handoff/Yu-Cao/results/`, not `artifacts/` — that directory ignores
`.txt` and `.png`, so files put there vanish without an error. `artifacts/*.log` is the
exception and is tracked; the sweep script already writes there.

---

## The rules, short

- **Do not edit `models/`, `training/`, `diagnostics/`, `config/`, `episodes/`,
  `adaptation/`, `domains/`, `scripts/cifar_table.py`, `scripts/phase_d_sweeps.sh`, or
  `evaluation/instruments.py`.** Two other people own those. Bugs go in your results file.
- You do not need `diagnostics/controls.py` and must not modify it. Stage 1 has always
  carried its own metrics. Your implementation of the mean-coordinate control will be the
  third written independently on this project, and agreement between the three is the only
  cheap check available on a quantity that has already been misread twice.
- **Deliver numbers, not conclusions.**
- Report absolute paired differences with 95 % intervals and the step count. Do not
  headline `task_specific_frac`: its denominator decays to zero and changes sign.
- Rebuild evaluation objects from the checkpoint's own stored config, never from current
  defaults.
- Add no dependency. torch and matplotlib are what you have.
