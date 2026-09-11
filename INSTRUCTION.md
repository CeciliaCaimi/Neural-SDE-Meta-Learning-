# Instruction — Jing Peng

Follow this in order. Your package is **training**: the parameterisation change the brief
proposes, and the comparison arm that decides whether this method's structure is worth
anything against ordinary conditioning.

The reasoning behind each step is in `handoff/Jing-Peng/README.md`. The values you may not
change are in `docs/PROTOCOL_CARD.md`. This file is the sequence.

---

## Step 1 — Clone and bootstrap

```bash
git clone -b jing-peng https://github.com/CeciliaCaimi/Neural-SDE-Meta-Learning-.git
cd Neural-SDE-Meta-Learning-
bash handoff/Jing-Peng/setup.sh
```

Put the `export CIFAR100_ROOT=...` line it prints into your shell profile. CIFAR-100 is
never committed to this repository and must not be copied into the working tree.

```bash
bash handoff/Jing-Peng/verify.sh
```

It reports how many GPUs you have — with more than one, step 5 parallelises. The preflight
must end with `preflight complete`; do not continue past a `FAIL`.

## Step 2 — Centred coordinates (about 2 hours)

Write the coordinate as `z_y = z̄ + δz_y` and let the shared predictor absorb the common
part. Four edits, roughly fifteen lines, and **no new network**:

```python
# 1. models/score_model.py
self.register_buffer("z_center", torch.zeros(self.k))
# in _prepare_z, immediately before h_eta:
if self.center_coords:
    zz = zz - self.z_center          # a buffer: no gradient path
# plus update_center(), an EMA at decay 0.99 under torch.no_grad

# 2. training/meta_train.py, after compute_coordinates (line 68)
if model.training:
    model.update_center(z_s, z_enc_t)

# 3. training/loop.py, EMA.__init__ -- EXCLUDE z_center from the shadow
# 4. tests/test_score_identity.py -- parameterise the identity assertion
```

Three things that must be decided rather than discovered:

- **The null control moves.** Under centring, `z = 0` is not a null; it is an off-centre
  perturbation with no interpretation. The null is `z = z̄`, which makes the basis term
  vanish exactly. Drop `loss_zero` from the centred panel rather than leave it to be
  misread. What remains is `delta_task = L(z̄) − L(z_y)` and the shuffled control, which
  is now the sharper of the two because it matches the coordinate's norm.
- **Do not delete the identity assertion** in `tests/test_score_identity.py`. With
  centring off it is `eps_hat(0) == eps_hat_0`; with centring on it is
  `eps_hat(z̄) == eps_hat_0`. Bit for bit, both ways.
- **`EMA.__init__` shadows every floating-point state-dict entry.** The noise-schedule
  buffers are constant so shadowing them is harmless; `z_center` drifts, so shadowing it
  would leave the diagnostics — which run under `ema.applied()` — reading a centre that
  training never used.

In the same edit, upgrade the diagnostics: 8 episodes → 24, one noise draw → 4, log
**per-episode** paired differences rather than only their mean, and add the fields
`delta_task` and `z_center_norm`. **Those exact names**: someone else's plotting script
reads them and the two of you are not talking.

Watch `z_center_norm` and coordinate spread. Centring introduces a gauge freedom — the
encoder can shift its own output mean to cancel the subtraction — and those two numbers
are what make it visible.

## Step 3 — Test both ways

```bash
python -m tests.test_score_identity
```

```bash
python -m tests.test_document_conformance
```

Both must pass with the centring flag on and off.

## Step 4 — The small pair (28 minutes each)

```bash
bash handoff/Jing-Peng/run_all.sh small
```

It checks that `runner.train` accepts `--center-coords` before starting anything, so a
missing flag costs you a second rather than the first run.

## Step 5 — The full pair (72 minutes each)

```bash
bash handoff/Jing-Peng/run_all.sh
```

Run these **whatever sign step 4 produced**: positive confirms it on the protocol
backbone, negative rules out a small-backbone artefact. On a multi-GPU instance the four
runs are independent — set `CUDA_VISIBLE_DEVICES=<i>` per run and start them together
instead of waiting on this script.

The two centring-**off** runs are not optional and are not merely a control. No run on
disk has ever logged the mean-coordinate panel, and the basis-usage and coordinate-spread
curves the brief asks for are exactly that.

> **One point where you may be able to save 1.7 GPU-hours.** Before launching, ask whether
> the generation test on the measurement side has finished. If it has, and the four
> coordinates turned out to be **indistinguishable**, run all four here — centring is the
> right response. If generation turned out to **differ**, the diagnostic was the problem
> rather than the representation: run only the two `_off` runs and leave the centring code
> in place unverified. If there is no answer yet, run all four. Everything in steps 2, 3
> and 6 is needed in every case.

## Step 6 — The comparison arm (about 3 hours to write, 2 to run)

The U-Net already conditions on time through feature-wise linear modulation, so a generic
latent enters the same path:

```python
# models/film_unet.py, registered as "film_unet"
temb = self.time_mlp(timestep_embedding(t, self.base_channels)) + self.z_mlp(z)

# baselines/film_conditioning.py -- a ScoreModel subclass with no basis term
class FiLMScoreModel(ScoreModel): ...

# runner/train_film.py -- use the seam that already exists
import models.film_unet
train(cfg, model_cls=FiLMScoreModel)
```

**Do not edit `training/loop.py` to add this arm**, even though you own that file. The
`model_cls` seam exists so the comparison arm and the centring change stay separable;
mixing them makes the matched comparison unverifiable. Keep centring off here, so this arm
aligns with `ctr128_off`.

```bash
bash handoff/Jing-Peng/run_all.sh film
```

Print both parameter counts — `z_mlp` against the basis head, each as a fraction of the
backbone — rather than asserting the budgets are equal.

## Step 7 — Hand back

```bash
cp checkpoints/ctr*_log.jsonl checkpoints/film128_log.jsonl handoff/Jing-Peng/results/
```

```bash
git add models training diagnostics config runner tests baselines handoff/Jing-Peng/results
git commit -m "E13 centred coordinates and E15 comparison arm"
git push origin jing-peng
```

The `*_log.jsonl` copies are the easiest thing to forget and the primary evidence:
`checkpoints/` is git-ignored, so the originals do not travel. Checkpoints are 430 MB
each — leave them on the server and say in your results file where they are. Check
`git status` for a stray `.pt` before committing.

---

## The rules, short

- **Do not edit `scripts/`, `evaluation/`, `episodes/`, `adaptation/`, `domains/`.** Two
  other people own those. Bugs go in your results file, not in a patch.
- **Add no network.** A wider transport, confidence gates, separate encoders and
  probabilistic latents are all ruled out. The change is a buffer and a subtraction.
- **Deliver numbers, not conclusions.**
- **Read nothing before the curve has levelled.** Quote the mean over the final fifth of
  training, report the slope over the last 10 000 steps, and state the step count.
- Report absolute paired differences with 95 % intervals. Do not headline
  `task_specific_frac`: its denominator decays to zero and changes sign.
- Every diagnostic on the **validation** split. `test` is untouched.
- Do not change any value in `docs/PROTOCOL_CARD.md`. The comparison between your two arms
  and the other packages is only meaningful because those values are identical everywhere.
