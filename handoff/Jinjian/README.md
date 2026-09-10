# Jinjian — Measurement

**Machine:** the local workstation, RTX 5080, 16 GB. Windows PowerShell 5.1.
**Budget:** about 5 hours of development, 1.5 hours of GPU.
**Branch:** `jinjian`

You have what nobody else has: the CIFAR-100 dataset and all three trained checkpoints,
which are 430 MB each and are not in git. Everything in this package reads them. No
training happens here.

Your package answers the first question of the supervisor's note — **does the coordinate
change what the model generates?** — and prepares the sweep harness for integration.

---

## What to build

| File | What it does |
|---|---|
| `scripts/posthoc_controls.py` | new — the four-way loss panel and the coordinate-loss profile, read off an existing checkpoint |
| `evaluation/instruments.py` | new — the measurement instruments for generated images |
| `scripts/gen_specificity.py` | new — generation from four coordinates on one initial noise, and the four metrics |
| `scripts/cifar_table.py` | yours to modify — add the crossover statistic |
| `scripts/phase_d_sweeps.sh` | yours to modify — unpin the hard-coded checkpoint |

**Do not touch** `models/`, `training/`, `diagnostics/`, `config/`. Those belong to Jing
Peng. In particular, do not import `diagnostics/controls.py` — write your own control
loop. He is independently upgrading that file, and two independent implementations of
this measurement are deliberate: it has already been misread twice on this project, and
a disagreement between two implementations is the only cheap way to catch a third.

---

## Task 1 — post-hoc controls (10 minutes of GPU)

The mean-coordinate substitution has never been run on the two full-width checkpoints.
`diagnostics/controls.py` computes it, but it was added at commit `ed98afb`, after every
CIFAR run had finished, so no log on disk contains it.

Write `scripts/posthoc_controls.py` so that it:

1. rebuilds the model, encoder and transport **from the checkpoint's own stored config**,
   never from current defaults — `scripts/cifar_table.py` shows the pattern, and the rule
   exists because scripts that rebuilt from the live config were silently wrong whenever
   the dimensions happened to match;
2. applies the exponential moving average weights;
3. evaluates, on 24 validation episodes with 4 noise draws each, the loss under the
   episode's own coordinate, one shared mean coordinate, another episode's coordinate,
   and zero — all four sharing one noising draw per episode;
4. prints per-episode paired differences and their 95 % intervals, not just means;
5. traces the coordinate-loss profile along two rays from the centroid,
   `L(z̄ + s·(z_y − z̄))` and `L(z̄ + s·(z_j − z̄))` for `s` in {0, ½, 1, 1½, 2}.

The profile is the cheap decisive part. Flat along the correct direction and rising along
wrong directions means the basis expresses distance from a centroid and nothing else.

**Checkpoint / split pairing is not advisory.** `cifar_ds3` was trained on three
relations and must be read against `cifar100_domainshift_c3.json`; the other two were
trained on one and must be read against `cifar100_domainshift.json`. The guard in
`cifar_table.py` enforces this and is correct.

## Task 2 — generation-level task specificity (1 hour of GPU)

`evaluation/instruments.py`:

- **Transformation statistics**, deterministic and instrument-free: mean absolute
  Laplacian response (blur suppresses it), per-image pixel standard deviation (contrast
  suppresses it), residual-noise estimate (additive noise raises it). One statistic per
  relation in the family.
- **A semantic instrument**: a small convolutional classifier over the 20 validation
  classes, trained on 550 images per class with 50 held out, and it must **report its own
  accuracy on those 50** so its verdict can be calibrated. The validation family contains
  leopard / lion / tiger and bowl / plate / cup, so report 4-way superclass accuracy
  alongside 20-way fine accuracy.

`scripts/gen_specificity.py`:

- 20 validation classes × four coordinates × 256 samples;
- DDIM at `eta = 0`, 50 steps, `x0` clipped to [−1, 1];
- one generator seed per sample index, **shared across all four coordinates**, so every
  comparison is paired on identical initial noise;
- split the 230-image target query pool in half: encode the coordinate from one half,
  score against the other. Without the split, every distance is optimistic.

Four measurements: the transformation statistic as a position on the clean-to-corrupted
axis; class consistency; distributional distance (sliced Wasserstein in pixel space and
energy MMD in the classifier's penultimate features); and **task-identification
accuracy** — assign each generated set to the class whose real half it is nearest to,
chance being 5 %. That last one is the headline, because unlike a ratio it stays readable
when the effect is small.

Also produce a paired visual grid: four rows sharing a column of initial noise, for three
episodes chosen *before* looking at the numbers.

> **Departure from the note, and say so in your results file.** It asks for the kernel
> Inception distance. Inception is a dependency this repository does not carry, and with
> 115 real images per class the estimate would be dominated by its own variance. Section 9
> of `ALGORITHM.md` already prescribes sliced Wasserstein or energy MMD for exactly this
> case. If the standard number is wanted for the paper it can be added at integration, on
> pooled samples rather than per class.

## Task 3 — sweep harness (30 minutes)

Change `CK=` at the head of `scripts/phase_d_sweeps.sh` to `CK=${1:-...}` so it takes a
checkpoint argument, and add to `scripts/cifar_table.py` one statistic it does not
currently print: **the value of `K_T` at which target-only overtakes transport**.
Coordinate space already places that crossover between 5 and 20; if the loss-space
crossover lands in the same interval, coordinate accuracy and denoising benefit are
connected at last.

Validate the harness on `ds_cap32_60k_step60000.pt`. That is a tool check, not a result —
the real sweep runs at integration, on whichever checkpoint passes.

---

## How to know you are finished

- Both 128-channel checkpoints have a `delta_task` with a 95 % interval and a step count.
- The coordinate-loss profile is plotted, both rays.
- The classifier's own held-out accuracy is printed next to its verdict on generated images.
- Task-identification accuracy exists for all four coordinates, with the chance interval.
- The visual grid has four rows on one column of noise.
- `phase_d_sweeps.sh <any checkpoint>` runs, and the crossover has a value.

## Deliver into `handoff/Jinjian/results/`

`E12a_controls.txt`, `E12_generation.txt`, `E14_harness_check.txt`,
`coord_profile.png`, `gen_grid.png`.

Not into `artifacts/` — it ignores `*.txt` and `*.png`, and your files would silently
fail to commit.

**Do not write a conclusion.** Deliver the numbers.
