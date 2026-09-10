# Protocol card — next-stage work packages

Three people are running the next stage of this project in parallel, from three
branches, **without coordinating with each other**. That only works if everything they
share is fixed in advance. This file is what they share. Read it before writing code,
and do not renegotiate any value in it on your own.

The experimental design behind each package is in the two planning documents; this file
carries only the parts that must be *identical* across all three.

---

## 1. Fixed values

| Item | Value | Why it cannot be chosen locally |
|---|---|---|
| Global seed | `global_seed = 12345` | Split assignment and episode sampling both derive from it |
| Split file | `artifacts/cifar100_domainshift_c3.json` | Fixes the task family; see §2 |
| Task family | `gaussian_blur`, `gaussian_noise`, `contrast`, severity 3 | The one arm where a coordinate signal is measurable at all (§4) |
| Backbone, full | `base_channels=128`, `channel_mult=(1,2,2)`, `num_res_blocks=2`, `attn_resolutions=(16,)` | Matched-comparison condition 1 |
| Backbone, small | `--base-channels 32` (drops to `(1,2)`, 1 block, no attention automatically) | Same, for the fast arm |
| Coordinate dimension | `k = 16` at 128 channels; `k = 32` at 32 channels | Matched-comparison condition 2 |
| Source evidence | `M_S = 64` during training | Matched-comparison condition 3 |
| Target evidence | `K_T` drawn from {1, 2, 5, 10, 20} | Matched-comparison condition 3 |
| Transport | stock `models/transport.py`, `hidden = 64`, do not widen | Matched-comparison condition 4 |
| Refinement budget | `J = 25`, `eta_z = 1e-2`, `beta_0 = 1.0`, `noise_batch = 32` | Matched-comparison condition 5 |
| Data discipline | every reading on **val**; `test` is touched once, for a final number | Matched-comparison condition 6 |
| Evaluation noise | one generator seed per episode, reused across strategies | Matched-comparison condition 7 |
| Optimiser | AdamW, `lr = 2e-4`, 500-step warmup, grad clip 1.0, EMA 0.999, bf16 | Unchanged from every result on record |

The seven "matched-comparison conditions" are the list in §5 of the supervisor's note.
They are the reason this file exists: the comparison between the structured
reverse-dynamics basis and generic latent conditioning is only meaningful if both arms
were trained under identical conditions, and the two arms are being built by different
people who are not talking to each other.

---

## 2. Split files, and the rule about them

| File | SHA-256 | Corruptions |
|---|---|---|
| `artifacts/cifar100_domainshift_c3.json` | `2ae9b81b639468fbbe5c6f087ed42c5d4eb924184f734988af00540768047708` | blur, noise, contrast |
| `artifacts/cifar100_domainshift.json` | `bed53927daac3c490114af48c37b5325b9a2803a5fa1a89304bb177d7d5c5d74` | blur only (historical) |

Verify before your first run:

```bash
python -c "import hashlib;print(hashlib.sha256(open('artifacts/cifar100_domainshift_c3.json','rb').read()).hexdigest())"
```

**Nobody rebuilds or overwrites a split file.** The builder now refuses to overwrite an
existing one, and the second file above is kept only because three checkpoints on disk
were trained against it. The two files carry *identical image-index streams* — the
corruption list does not enter index assignment — so they differ only in which
relations an episode may draw.

Pass the file explicitly on every command rather than relying on the config default:

```bash
python -m runner.train --scheme domainshift --domainshift-path artifacts/cifar100_domainshift_c3.json ...
```

```bash
python scripts/cifar_table.py <ckpt> --domainshift-path artifacts/cifar100_domainshift_c3.json ...
```

A checkpoint trained on one file and evaluated against the other is refused by the
corruption-count guard in `scripts/cifar_table.py`. That guard is correct; do not
weaken it.

**CIFAR-100 itself is never committed.** It is referenced, never copied, and the
directory it sits in on the machine of record also holds clinical data that must not
reach GitHub. On any other machine, point at your own copy with an environment variable
rather than editing a path in the code:

```bash
export CIFAR100_ROOT=/home/you/data/cifar-100-python
```

Set it before the process starts; `domains/cifar100.py` reads it at import. Stage 1 needs
no dataset at all — its Gaussian mixtures are generated from the seed.

---

## 3. How to report a number

Four rules, each of which was earned by getting it wrong on this project.

**Pair everything, and report the difference.** Between-episode variance is roughly
sixteen times the effect being measured; the standard deviation of the paired difference
is sixty-five times smaller than that of the raw loss. All controls already share one
noising draw inside `ScoreModel.eps_hat_many` — keep it that way. Never present a raw
loss curve as evidence about a task-specificity effect.

**Read nothing before the curve has levelled.** Quote the mean over the final fifth of
training, and state the slope over the last 10 000 steps. Always print the step count
beside the number. A 300-step probe was once reported as a fix and had collapsed by step
1 000.

**Prefer an absolute paired difference with a 95 % interval to a ratio.** Do not report
`task_specific_frac` as the headline: its denominator (`gain_vs_zero`) decays towards
zero and changes sign, so the ratio becomes unstable exactly where the reading matters.
Report `delta_task` in loss units, with its interval.

**Rebuild evaluation objects from the checkpoint's own configuration**, never from the
current defaults. `scripts/cifar_table.py` shows the pattern. Scripts that rebuilt from
the live config crashed loudly when dimensions differed and were silently wrong when
they happened to match.

---

## 4. Facts you may need, so you do not have to re-derive them

Read out of `checkpoints/*_log.jsonl` on 10 September 2026. Quote these rather than
recomputing them, and do not treat them as conclusions.

- **The shuffled control is not zero on the three-corruption run.** Over the last ten
  diagnostic points of `cifar_ds3` (steps 41k–50k, 128 channels, `k=16`), substituting
  *another episode's* coordinate costs **+0.0029** in denoising loss, while deleting the
  coordinate entirely costs **+0.0016**. On every other run this quantity is +0.00003.
  Working hypothesis: the coordinate carries transformation identity, not class identity.
- **Coordinates are well spread.** At step 49 000 of `cifar_ds3`, `z_spread_rel = 1.208`
  — the between-episode spread is of the same order as the coordinate norm itself. A
  centred parameterisation therefore does not remove most of the vector.
- **Diagnostic noise.** Over steps 31k–50k the standard deviation of `loss_correct`
  between diagnostic points is **0.0241**; of the paired difference, **0.00037**.
- **No run on disk logs the mean-coordinate control.** It was added at commit `ed98afb`,
  after every CIFAR run had finished, so none of the eighteen logs contains
  `loss_mean_z` or `task_specific_frac`. Any curve of those quantities requires a new run.

Checkpoints available to read (all trained before the control existed):

| File | Step | Backbone | `k` | Relations | Split file |
|---|---|---|---|---|---|
| `cifar_ds3_step50000.pt` | 50 000 | 128 ch | 16 | 3 | `_c3` |
| `cifar_ds_blur_step50000.pt` | 50 000 | 128 ch | 16 | 1 | historical |
| `ds_cap32_60k_step60000.pt` | 60 000 | 32 ch | 32 | 1 | historical |

The last column is not advisory. Only `cifar_ds3` may be read against
`cifar100_domainshift_c3.json`; the other two were trained on a single relation and must
be read against `cifar100_domainshift.json`, or the corruption-count guard fires. New
training runs all use `_c3`.

---

## 5. Who runs what, and file ownership

The allocation follows the hardware. Full briefs are in `handoff/<Name>/README.md`.

| | Machine | Package | Branch |
|---|---|---|---|
| **Jinjian** | local workstation, RTX 5080 | Measurement — E12a controls, E12 generation test, E14 harness | `jinjian` |
| **Jing Peng** | AWS GPU server | Training — E13 centred coordinates, E15 generic-latent arm | `jing-peng` |
| **Yu Cao** | Mac, CPU only | Positive control — the stage-1 panel and the task-distance sweep | `yu-cao` |

One owner per path. Check your column before editing anything.

| Path | Jinjian | Jing Peng | Yu Cao |
|---|---|---|---|
| `models/score_model.py` | — | **owns** | — |
| `models/film_unet.py` (new) | — | **owns** | — |
| `training/loop.py`, `training/meta_train.py` | — | **owns** | — |
| `diagnostics/controls.py` | — | **owns** | — |
| `config/base_config.py` | — | **owns** | — |
| `baselines/` (new) | — | **owns** | — |
| `runner/train.py`, `runner/train_film.py` (new) | — | **owns** | — |
| `tests/test_score_identity.py` | — | **owns** | — |
| `evaluation/instruments.py` (new) | **owns** | — | — |
| `scripts/posthoc_controls.py`, `scripts/gen_specificity.py` (new) | **owns** | — | — |
| `scripts/cifar_table.py`, `scripts/phase_d_sweeps.sh` | **owns** | — | — |
| `scripts/stage1_panel.py`, `scripts/plot_panel.py` (new) | — | — | **owns** |
| `scripts/stage1_missing_metrics.py`, `scripts/run_relatedness_sweep.sh` | — | — | **owns** |
| `runner/stage1_gmm.py` | — | — | **owns** |
| `artifacts/*.json` | read only | read only | read only |
| existing `checkpoints/*` | **reads** | read only | read only |

Three consequences worth stating explicitly.

**Nobody imports `diagnostics/controls.py` except its owner.** Jinjian writes his own
control loop in `scripts/posthoc_controls.py`; Yu Cao writes a third for stage 1. That is
deliberate, not waste: this project has already produced two confident wrong readings of
exactly this quantity, and disagreement between independent implementations is the only
cheap way to catch a third.

**The FiLM arm does not edit `training/loop.py`.** `build()` and `train()` take
`model_cls`, so a `ScoreModel` subclass reaches the loop without touching it:

```python
import models.film_unet          # registers the backbone
train(cfg, model_cls=FiLMScoreModel)
```

**Log field names are an interface.** Jing Peng's runs must emit `delta_task` and
`z_center_norm` under exactly those names, because Yu Cao's `scripts/plot_panel.py`
consumes them and the two of them are not talking. That plotter must in turn survive both
fields being absent, since no log currently on disk has either.

Name new checkpoints with your own `--run-name` prefix so three people's outputs never
collide in `checkpoints/`.

Branches are all cut from the commit that introduces this file. Do not rebase onto each
other.

---

## 6. What none of you does

**Do not draw the conclusion.** Deliver the numbers. Whether the generation test
separates the coordinates, and whether `delta_task` clears zero, are decided at
integration, against the criteria written above — which are fixed *before* anyone sees a
result, so that nobody can adjust a threshold after the fact.

**Do not run the final deployment sweep.** It happens at integration, on whichever
checkpoint passes, and takes about ninety minutes.

**Do not attempt the fixes ruled out in §7 of the supervisor's note**: a larger `k`,
another backbone size, a wider transport network, confidence gates, separate
source/target encoders, probabilistic latent machinery, or anything
healthcare-specific. The first two are not merely discouraged, they are already done and
negative: `k` in {16, 32, 64} decays to 0.0007 / 0.0005 / 0.0004, and shrinking the
backbone raises apparent gain 725-fold while the mean-coordinate substitution shows none
of it is task-specific. Repeating either would cost a day to reproduce a number already
in `artifacts/`.

**Do not scale to DomainNet.** Two conditions must hold first, and neither has been
tested yet.
