# Instruction — Jinjian

Follow this in order. Your package is **measurement**: it reads the three trained
checkpoints and asks whether the task coordinate changes what the model generates.
Nothing here trains a model.

The reasoning behind each step is in `handoff/Jinjian/README.md`. The values you may not
change are in `docs/PROTOCOL_CARD.md`. This file is the sequence.

---

## Step 1 — Get on your branch

Work in the checkout you already have. Do not clone a fresh one: the virtual
environment, CIFAR-100 and the three checkpoints all live there and none of them travel
through git.

```bash
git fetch origin
git checkout jinjian
bash handoff/Jinjian/verify.sh
```

The preflight must end with `preflight complete`. If it stops, it prints the one thing
to do about it. Do not continue past a `FAIL`.

## Step 2 — Post-hoc controls (about 1 hour to write, 10 minutes to run)

Write `scripts/posthoc_controls.py`. It must:

1. rebuild the model, encoder and transport **from the checkpoint's own stored config**,
   never from `BaseConfig()` defaults — `scripts/cifar_table.py` lines 84–96 show the
   pattern;
2. apply the EMA weights;
3. evaluate, over 24 validation episodes with 4 noise draws each, the denoising loss
   under this episode's own coordinate, one shared mean coordinate, another episode's
   coordinate, and zero — **all four sharing one noising draw**, which
   `ScoreModel.eps_hat_many` gives you in a single call;
4. print per-episode paired differences with 95 % intervals, not only their means;
5. trace the coordinate-loss profile along two rays from the centroid,
   `L(z̄ + s·(z_own − z̄))` and `L(z̄ + s·(z_j − z̄))` for `s ∈ {0, ½, 1, 1½, 2}`.

Then run it on both full-width checkpoints:

```bash
.venv/Scripts/python.exe scripts/posthoc_controls.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --n-episodes 24 --n-noise 4
```

```bash
.venv/Scripts/python.exe scripts/posthoc_controls.py checkpoints/cifar_ds_blur_step50000.pt --domainshift-path artifacts/cifar100_domainshift.json --split val --n-episodes 24 --n-noise 4
```

**The two split files are different and are not interchangeable.** `cifar_ds3` was
trained on three relations, `cifar_ds_blur` on one. Swapping them trips the
corruption-count guard, which is correct.

What you are looking for: the mean-coordinate substitution has never been run on either
of these checkpoints, because the control was added to the code after every CIFAR run had
already finished. A flat profile along the correct ray and a rising one along wrong rays
would mean the basis expresses distance from a centroid and nothing else.

## Step 3 — The generation test (about 4 hours to write, 1 hour to run)

`evaluation/instruments.py`:

- transformation statistics, deterministic and needing no trained instrument: mean
  absolute Laplacian response, per-image pixel standard deviation, residual-noise
  estimate;
- a semantic instrument: a small classifier over the 20 validation classes, trained on
  550 images per class with 50 held out, which **reports its own accuracy on those 50**
  so its verdict can be calibrated. Report 4-way superclass accuracy alongside 20-way,
  because the validation family contains leopard / lion / tiger and bowl / plate / cup.

`scripts/gen_specificity.py`: 20 validation classes × four coordinates × 256 samples,
DDIM at `eta = 0` over 50 steps, `x0` clipped to [−1, 1].

Three things that are easy to get wrong and fatal to the result:

- **one generator seed per sample index, shared across all four coordinates**, so every
  comparison sits on identical initial noise;
- **split the 230-image target query pool in half** — encode the coordinate from one
  half, score against the other. Without the split every distance is optimistic;
- choose the three episodes for the visual grid **before** looking at any number.

Four measurements: the transformation statistic as a position on the clean-to-corrupted
axis; class consistency; distributional distance (sliced Wasserstein plus energy MMD);
and task-identification accuracy, where chance is 5 %. The last is the headline.

State in your results file that sliced Wasserstein and energy MMD stand in for the
kernel Inception distance the brief asks for, and why: Inception is a dependency this
repository does not carry, and with 115 real images per class the estimate would be
dominated by its own variance. `ALGORITHM.md` §9 already prescribes these two.

## Step 4 — The sweep harness (about 30 minutes)

Change `CK=` at the head of `scripts/phase_d_sweeps.sh` to `CK=${1:-...}` so it takes a
checkpoint argument, and add to `scripts/cifar_table.py` one statistic it does not print
today: the value of `K_T` at which target-only overtakes transport.

## Step 5 — Run everything

```bash
bash handoff/Jinjian/run_all.sh
```

It checks, before spending any time, that your three scripts exist and that
`phase_d_sweeps.sh` no longer has its checkpoint written in. A missing piece stops it
with the name of the piece.

## Step 6 — Hand back

```bash
git add scripts evaluation handoff/Jinjian/results
git commit -m "E12a and E12: measurement results"
git push origin jinjian
```

Deliverables go in `handoff/Jinjian/results/`, not `artifacts/` — that directory ignores
`.txt` and `.png`, so files put there vanish without an error.

---

## The rules, short

- **Do not edit `models/`, `training/`, `diagnostics/`, `config/`.** Jing Peng owns them.
  If you find a bug there, write it in your results file and carry on.
- **Do not import `diagnostics/controls.py`.** Write your own control loop. The
  duplication is deliberate: this measurement has been misread twice on this project, and
  two independent implementations are the cheap way to catch a third.
- **Deliver numbers, not conclusions.** Whether the coordinate works is decided at
  integration, against thresholds fixed before any result was seen.
- **State the step count beside every number**, and quote the mean over the final fifth
  of training where a curve is involved.
- Report absolute paired differences with intervals. Do not headline
  `task_specific_frac`: its denominator decays to zero and changes sign.
- Add no dependency — no torchvision, no Inception, no torchmetrics — without saying so
  explicitly in the results file.

## If something stops

The preflight and the driver both fail with a sentence telling you what to do. If you hit
something they do not cover, write it down rather than working around it by editing a file
that is not yours.
