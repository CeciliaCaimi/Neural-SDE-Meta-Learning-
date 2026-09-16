# Final experiment plan — execution order

Source: David's *Experimental Plan Update*, 14 September 2026. This file turns that note
into an ordered sequence. Nothing here proposes new science; where the note says
"evaluate X", this file says which file to run, what it writes, and how to know the step
is finished.

**The implementation is frozen.** No architectural change is in scope (see §13 of the
note, repeated in [Excluded work](#excluded-work) below). Every item is either an
evaluation, a sweep, or a new dataset.

**The decisive criterion is generation-level source dependence.** The denoising quantity
`delta_task` is demoted to a supporting diagnostic; it is no longer a gate on anything.
This is a change from the previous stage's plan, where it was gate 2.

**Notation.** `K_T` is the number of target-domain examples available at adaptation time.
`M_S` is the number of source-domain images the encoder sees when producing the source
coordinate `z_S`. `delta_task` is the paired loss difference between a task's own
coordinate and the mean coordinate shared by all tasks. FSIG is few-shot image
generation. All commands assume the repository root as the working directory and the
project interpreter `.venv/Scripts/python.exe`.

---

## Stage 0 — Preconditions (do these first; everything else depends on them)

### 0.1 Integrate the three branches

The measurement scripts and the FiLM (feature-wise linear modulation) arm currently live
on different branches, so no cross-package measurement can run. Merge `jinjian` and
`jing-peng` into the baseline, then branch from the result.

```bash
git fetch origin
```

```bash
git checkout main
```

```bash
git merge origin/jinjian
```

```bash
git merge origin/jing-peng
```

**Done when:** `scripts/gen_strategies.py`, `models/film_unet.py` and
`baselines/film_conditioning.py` all exist in one tree and
`.venv/Scripts/python.exe -m pytest tests -q` passes.

Every command in this file is written to run identically in PowerShell and in a POSIX
shell: one command per block, no line continuations, no shell-specific flags.

### 0.2 Verify the split file before the first run

```bash
.venv/Scripts/python.exe -c "import hashlib;print(hashlib.sha256(open('artifacts/cifar100_domainshift_c3.json','rb').read()).hexdigest())"
```

Must equal `2ae9b81b639468fbbe5c6f087ed42c5d4eb924184f734988af00540768047708`. Pass
`--domainshift-path artifacts/cifar100_domainshift_c3.json` explicitly on every command
below; never rebuild or overwrite a split file.

### 0.3 Recover or retrain the FiLM checkpoint

`checkpoints/` is not tracked by git, so `film128` did not travel with the branch. Either
obtain `film128_step50000.pt` from the machine that trained it, or retrain:

```bash
.venv/Scripts/python.exe -m runner.train_film --scheme domainshift --domainshift-path artifacts/cifar100_domainshift_c3.json --run-name film128 --k 16 --steps 50000
```

**Done when:** `checkpoints/film128_step50000.pt` exists and its stored configuration
reports `base_channels=128`, `k=16`.

### 0.4 Add the missing evaluation primitives

Five small pieces of code are required by later steps and do not exist yet. Write them
once, here, rather than inside each experiment.

| Piece | Where | Note |
|---|---|---|
| `--m-source N` on `scripts/gen_strategies.py` | pass through to `DomainShiftLoader(enc_source_images=N)` | `scripts/cifar_table.py:127` already does exactly this; copy the pattern |
| **Nested** source subsampling | `episodes/domainshift.py:163` | `_sub` currently calls `rng.choice` without replacement, so each `M_S` draws a *different* set. For the `M_S` sweep take the prefix `pool[:n]` instead, which is nested by construction. Target support already uses the prefix `[:k]`, so `K_T` is nested today |
| Relation-only transport condition | `adaptation/coordinate.py` | Predict the target coordinate from the relation descriptor `c` alone, with `z_S` zeroed. The relation embedding already exists in `models/transport.py` |
| Within-relation shuffled source | `scripts/gen_strategies.py` | Swap `z_S` with another episode's `z_S` **drawn from the same relation**, not from any episode |
| KID (kernel inception distance) | `evaluation/metrics_analytic.py` | `sliced_wasserstein` and `energy_mmd` already exist there and are already used; KID is the only metric in the note that is missing. Report it only where sample counts support it |

**Done when:** each piece has a test or a smoke run, and `--m-source 16` visibly changes
a printed number.

### 0.5 Create the output directory

Results go under `handoff/`, never under `artifacts/`: `.gitignore` excludes `*.txt`,
`*.png` and `*.out` inside `artifacts/`, so files written there vanish without an error.

```bash
.venv/Scripts/python.exe -c "import os;os.makedirs('handoff/results',exist_ok=True)"
```

---

## Stage A — CIFAR-100, generation-level evidence

All four experiments in this stage share: the same held-out episodes, the same
target-query sets, the same initial diffusion noise seeds across conditions, and the
validation split. The test split is touched **once**, at the very end, for the final
number.

### A1 — Source-dependence controls (note §1)

Four conditions on the three-transformation setup: correct `z_S`, within-relation
shuffled `z_S`, mean source coordinate `z_S_bar`, and relation-only.

```bash
.venv/Scripts/python.exe scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 5 20 --n-classes 8 --grid-out handoff/results/A1_grid.png
```

Report: transformation / target-domain consistency, semantic-class consistency, sliced
Wasserstein or energy maximum mean discrepancy as the headline distributional metric,
KID only where sample counts allow, and absolute paired differences with 95 % confidence
intervals.

**Success criterion:** the full method beats shuffled-source, mean-source and
relation-only at generation time. **If relation-only matches the full method, say so
explicitly and narrow the source-informed claim** — that outcome is reportable, not a
failure.

**Output:** `handoff/results/A1_source_dependence.txt`, `A1_grid.png`.

### A2 — Additive basis against FiLM, at generation (note §2)

Extends E15 from denoising loss to generation. Matched on backbone, latent dimension,
transport capacity, support sets, training budget, generation seeds and evaluation
protocol.

```bash
.venv/Scripts/python.exe scripts/gen_specificity.py checkpoints/film128_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --grid-out handoff/results/A2_film_grid.png
```

```bash
.venv/Scripts/python.exe scripts/gen_strategies.py checkpoints/film128_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 5 20 --grid-out handoff/results/A2_film_strategies.png
```

Then run the identical pair on `cifar_ds3_step50000.pt` if A1 has not already produced
it, and tabulate side by side.

**The question:** does the additive reverse-dynamics basis transfer better at generation
time, **especially at low `K_T`**? The denoising-loss comparison was mixed — the basis
reached the better absolute loss while FiLM leaned on its coordinate far more — so this
step decides it.

**Output:** `handoff/results/A2_basis_vs_film.txt`, two grids.

### A3 — Target-scarcity sweep (note §3)

```bash
.venv/Scripts/python.exe scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 2 5 10 20 --n-classes 8 --n-samples 96
```

Four strategies: target-only, reuse `z_S`, transport only, transport + refinement. Same
episodes and generation seeds at every `K_T`.

**Primary output:** generation quality against `K_T`, plotted. **Explicitly record the
value or interval of `K_T` at which target-only catches up with or overtakes transport.**
Sixty episodes per `K_T` is the minimum that has produced stable readings on this
project; twelve produced two confidently wrong answers.

**Output:** `handoff/results/A3_kt_curve.txt`, `A3_kt_curve.png`.

### A4 — Source-abundance sweep (note §4)

Fix `K_T = 1`; sweep `M_S` over {16, 32, 64, 128, 256, 300}. Requires the nested
subsampling from step 0.4 — without it, set identity is confounded with set size.

```bash
.venv/Scripts/python.exe scripts/gen_strategies.py checkpoints/cifar_ds3_step50000.pt --domainshift-path artifacts/cifar100_domainshift_c3.json --split val --k-shots 1 --m-source 16
```

Repeat for each value of `M_S`. The upper bound 300 is the source pool size fixed in
`episodes/domainshift.py:36`; do not exceed it.

**Desired result:** improvement with increasing source evidence, then saturation.

**Output:** `handoff/results/A4_ms_curve.txt`, `A4_ms_curve.png`.

### A5 — Transport-structure ablation (note §5)

Four maps from the source coordinate to the predicted target coordinate, under the same
generation protocol:

1. direct reuse, `z_T = z_S`;
2. constant relation vector, `z_T = z_S + delta`;
3. linear map, `z_T = A z_S + b`;
4. the current nonlinear residual map, `z_T = z_S + Delta_gamma(z_S, c)`.

Variants 2 and 3 are new and belong beside `models/transport.py`; keep them in coordinate
space only, and do not widen the hidden layer past 64.

**Report:** whether nonlinear transport materially improves on direct reuse, on a
constant relation vector, or on a linear map. A negative answer here is a result and
simplifies the method section.

**Output:** `handoff/results/A5_transport_ablation.txt`.

---

## Stage B — Gaussian-mixture positive control (note §6)

Runs on a laptop, needs no dataset and no GPU, and **must not delay Stage A or C**. It is
supporting calibration only.

1. Four-way panel on the stage-1 checkpoint: own coordinate, mean coordinate, shuffled
   coordinate, zero.
2. Coordinate-loss profile along `z_bar + s (z_own - z_bar)` for
   `s` in {0, 0.5, 1, 1.5, 2}. A clear minimum at `s = 1` is expected here; on images the
   profile is flat, and the contrast is the point.
3. Retain the existing relatedness / task-distance sweep if it is available.

Take the mean **across tasks, not across the batch** — every query point inside one task
already shares a coordinate, so averaging along the batch dimension is a no-op and is the
easiest way to get a wrong answer here.

**Output:** `handoff/Yu-Cao/results/`, plus `scripts/plot_panel.py` for shared use.

---

## Stage C — Demographic scarcity experiment (note §8, §9, §10)

The preferred real-world test, because it supplies **repeated instances of one population
relation across many semantic tasks** — exactly what FSIG does not. Start C0 on day one;
it gates everything else in this stage.

### C0 — Same-day feasibility check (hard gate)

Primary dataset: **Fitzpatrick17k**, with skin condition as the semantic task and skin
phenotype group as the source/target population split.

1. Confirm the complete image set is actually retrievable.
2. Build a condition-by-phenotype-group count table.
3. Choose the phenotype grouping from the observed counts, not in advance.
4. Keep only conditions with enough source and target examples for train, validation,
   test and a held-out target query set. Starting threshold, to be revised from the real
   counts: `N_S >= 30–40`, `N_T >= 15–20`.

**If access or counts fail, switch the same day** to **NIH ChestX-ray14** with thoracic
finding as the semantic task and age group as the population split: build the
finding-by-age-bin table, prefer single-positive-finding cases, enforce **patient-level**
separation, and pick age bins that give a source-rich / target-sparse relation while
leaving enough target query images.

**Data handling.** Neither dataset may be committed. Reference it by an environment
variable read at import, exactly as `domains/cifar100.py` reads `CIFAR100_ROOT`. The
repository root must not widen.

**Output:** `handoff/results/C0_feasibility.txt` containing both count tables and the
chosen grouping.

### C1 — Domain module and splits

Add `domains/<dataset>.py` and a split builder that refuses to overwrite an existing
split file, following `runner/build_cifar100_splits.py`. **Split the semantic tasks
themselves** into train / validation / test — this is what makes the relation, rather
than the task, the object of generalisation.

### C2 — Frozen evaluation predictors

Before any headline number: a frozen semantic / diagnostic consistency predictor, and a
demographic or phenotype consistency predictor where a suitable one exists. Each must
report its own held-out ceiling beside every verdict it is used for.

### C3 — Optional sanity check

On a small number of tasks, verify that the model produces a measurable target-population
shift **when target evidence is abundant**. This is a sanity check on the pipeline, not a
gate, and it is not motivated by the FFHQ result.

### C4 — The scarcity sweep

Abundant source-population examples; `K_T` in {1, 2, 5, 10, 20}. Arms:

1. target only;
2. source reuse;
3. population / relation only;
4. **wrong-task source + relation**;
5. correct-task source + Meta-Diffusion transport;
6. FiLM + transport, if time allows.

Metrics: distance to the real held-out target-population distribution; semantic /
diagnostic consistency; demographic consistency; sliced Wasserstein or energy maximum
mean discrepancy as the main small-sample metric; KID only where statistically
meaningful; absolute paired differences with 95 % confidence intervals.

**Report target-data equivalence**, for example *Meta-Diffusion at `K_T = 2` matches
target-only at `K_T = 10`*. This is the headline form, and it is preferable to quoting a
small metric improvement.

**Output:** `handoff/results/C4_demographic_scarcity.txt`, plus the curve.

### C5 — Downstream augmentation (optional)

Run **only if C4 is clearly positive.** Build an imbalanced training set with abundant
source-population data and few target-population examples, then compare: real imbalanced
data only; target oversampling; target-only synthetic augmentation; Meta-Diffusion
synthetic augmentation. Evaluate on real held-out target-population images and also
report source-population performance. Use one simple fixed downstream model; do not tune
the classifier.

---

## Stage D — FFHQ / FSIG disposition (note §7)

**No new runs.** The negative result stands and is retained, reinterpreted as a
task-family mismatch and stress test rather than as evidence against the mechanism: the
standard FSIG setup offers a handful of highly heterogeneous target domains (babies,
sunglasses, MetFaces, emojis, sketches) instead of many repeated instances of one
reusable source-to-target relation.

- `K_T = 10` is the only benchmark-valid FSIG setting; report that one.
- Treat `K_T > 10` runs as internal expressivity diagnostics only.
- Do not use the `K_T` = 100, 500, 1000 runs as headline FSIG comparisons.
- Do not change the architecture in response to it, and do not spend further time trying
  to make FFHQ-FSIG the main positive experiment.

Reported as a limitation: **few heterogeneous target domains implies weak reusable
task-family structure.**

---

## Cross-cutting requirements

### Baselines (note §11)

Minimum set for any headline comparison: target-only conditioning; direct source reuse;
relation-only prediction; FiLM + matched transport; additive basis + transport; transport
with and without refinement; LoRA (low-rank adaptation) or adapter adaptation if
available; full or partial fine-tuning with a validation-selected optimisation budget.

`full_ft` and `full_ft_oracle` already exist in `adaptation/coordinate.py:35`. LoRA or an
adapter does not and would be new work — schedule it only after Stage A is complete.
**Do not use equal optimisation steps as the only fairness criterion for full
fine-tuning.**

### Evaluation discipline (note §12)

- Pair every condition on the same episodes, target-query samples and generation seeds.
- Report absolute paired differences with 95 % confidence intervals.
- Read a training-dependent quantity only after its curve has levelled; quote the mean
  over the final fifth of training and the slope over the final window, and always print
  the step count beside the number.
- Never headline a ratio whose denominator approaches zero (`task_specific_frac`).
- Never present a raw loss curve as evidence about a task-specificity effect: between
  diagnostic points the raw loss varies roughly sixty-five times more than the paired
  difference being measured.

### Excluded work (note §13)

For this submission, do not: continue coordinate-centring experiments (a clean negative
already); raise `k` as a primary intervention; add gating, probabilistic-latent or
separate source/target modules; run broad head-architecture searches unless generation
quality blocks evaluation; add a second demographic dataset before the first is complete;
promote refinement into the headline method unless it shows a reliable generation
benefit; or spend further time on FFHQ-FSIG.

Two consequences worth stating plainly:

- **dSprites is out of scope.** The role it was meant to play — a task variable that
  cannot be read off a single target image — is now played by the demographic experiment.
- **The severity sweep is out of scope** unless it is needed to interpret a Stage A
  result. The flag remains wired at `runner/train.py:37`.

---

## Final deliverables (note §14)

1. Gaussian-mixture mechanism result and the cheap positive-control calibration — Stage B
2. CIFAR source-dependence controls — A1
3. CIFAR additive basis against FiLM, at generation — A2
4. `K_T` scarcity curve, with the crossing point named — A3
5. `M_S` source-abundance curve — A4
6. Transport-structure ablation — A5
7. One demographic scarcity experiment — C4
8. Paired 95 % confidence intervals for every headline comparison — all stages
9. Optional downstream demographic augmentation — C5
10. Optional FFHQ-FSIG stress test at the valid `K_T = 10` setting — Stage D

**The main claim is supported only if** correct source task **plus** learned
source-to-target relation outperforms **both** relation-only transfer **and** target-only
few-shot modelling, in the low-target-data regime. Any other outcome is reported as what
it is.

---

## Ordering and parallelism

```
day 1        0.1 merge  ->  0.2 checksum  ->  0.4 primitives      | C0 feasibility (start same day)
day 1-2      0.3 film128 retrain (background, ~2 h GPU)           | C1 splits
day 2-3      A1 source dependence  ->  A3 K_T curve               | C2 frozen predictors
day 3-4      A2 basis vs FiLM  ->  A4 M_S curve                   | C3 sanity check
day 4-6      A5 transport ablation                                | C4 scarcity sweep
day 6-7      write-up, test-split final number, C5 if C4 is clean | Stage B lands whenever
```

Three dependencies are real; the rest is free:

- **0.1 blocks A2** — the FiLM model code and the generation scripts are on different
  branches.
- **0.4 blocks A4** — without nested source sets the sweep measures the wrong thing.
- **C0 blocks all of Stage C** — and if it fails, it must fail on day one, not in week
  two.

Stage B is independent of everything and must not be allowed to hold anything up. Stage D
requires no compute at all.
