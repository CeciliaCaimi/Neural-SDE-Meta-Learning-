# Jing Peng — Training

**Machine:** AWS GPU server.
**Budget:** about 5.5 hours of development, 5.5 hours of GPU — but they parallelise, so
the wall clock is as short as your instance count allows.
**Branch:** `jing-peng`

You have the compute, so you have the two arms that need it: the parameterisation change
that the supervisor's note proposes, and the comparison arm that decides whether the
method's structure is worth anything against ordinary conditioning.

Everything you need travels in the clone except CIFAR-100 itself, which is deliberately
never committed. `setup.sh` fetches it and points the code at it.

---

## Task 1 — centred coordinates (E13)

Write the task coordinate as `z_y = z̄ + δz_y` and let the shared predictor absorb the
common part:

```
eps_hat_y(x,t) = [ eps_hat_0(x,t) + B(x,t) z̄ ] + B(x,t) δz_y
```

Both sides describe the same function class. What changes is where the shared offset is
allowed to live, and therefore what the diagnostics mean: today `r_basis` is 0.52 on one
checkpoint while none of it is task-specific, because the ratio is dominated by `B z̄`.
Once the offset is absorbed, basis usage, the gain over the null, and the task-specific
effect become the same quantity, and a constant offset dressed as adaptation becomes
inexpressible.

Four edits, none of which adds a network:

```python
# 1. models/score_model.py
self.register_buffer("z_center", torch.zeros(self.k))
# in _prepare_z, immediately before h_eta:
if self.center_coords:
    zz = zz - self.z_center            # a buffer: no gradient path
# plus update_center(), an EMA at decay 0.99 under torch.no_grad

# 2. training/meta_train.py, after compute_coordinates (line 68)
if model.training:
    model.update_center(z_s, z_enc_t)

# 3. training/loop.py, EMA.__init__ -- EXCLUDE z_center from the shadow
# 4. tests/test_score_identity.py -- parameterise the identity assertion
```

Three consequences, each of which must be decided rather than discovered:

**The null control moves.** Under centring, `z = 0` is no longer a null; it is an
off-centre perturbation with no interpretation. The null is `z = z̄`, which makes the
basis term vanish exactly. Drop `loss_zero` from the centred panel rather than leave it
to be misread. What remains is `delta_task = L(z̄) − L(z_y)` and the shuffled control,
which is now the sharper of the two because it matches the coordinate's norm.

**`tests/test_score_identity.py` asserts `eps_hat(0) == eps_hat_0` bit for bit.** Under
centring the identity becomes `eps_hat(z̄) == eps_hat_0`. Parameterise the assertion on
the flag; do not delete it, because it is what gives the null control its meaning in
either parameterisation.

**`EMA.__init__` shadows every floating-point entry of the state dictionary.** Today that
is harmless — the four noise-schedule buffers are constant, so their moving average
equals themselves. `z_center` drifts, so shadowing it would leave the diagnostics, which
run under `ema.applied()`, reading a doubly smoothed centre that training never used.

**The one thing that could go wrong:** centring introduces a gauge freedom, since the
encoder can shift its own output mean to cancel the subtraction. Two guards. The
coordinate regulariser still pins the scale, and logging `|z_center|` beside
`z_spread_rel` makes the drift visible. There is also reason to expect the signal to
survive: at step 49 000 of `cifar_ds3` the relative spread of coordinates across episodes
is 1.208, so the between-task part is of the same order as the coordinate itself.

### Diagnostic upgrade, in the same edit

- 8 episodes → 24, and 1 noise draw → 4 (`training/loop.py:216`);
- log **per-episode** paired differences, not only their mean, so an interval can be
  drawn as a band;
- add the fields `delta_task` (keeping `gain_vs_mean_z` as its alias) and `z_center_norm`
  — those exact names, because someone else's plotting script reads them;
- compute the substitution coordinate from a running mean over training episodes rather
  than from the mean of the current diagnostic batch.

### Runs

| Name | Backbone | k | Centring | Steps | Time |
|---|---|---|---|---|---|
| `ctr32_off` | 32 ch | 32 | off | 60 000 | ~28 min on an RTX 5080 |
| `ctr32_on` | 32 ch | 32 | **on** | 60 000 | ~28 min |
| `ctr128_off` | 128 ch | 16 | off | 50 000 | ~72 min |
| `ctr128_on` | 128 ch | 16 | **on** | 50 000 | ~72 min |

The `_off` runs are not optional and are not merely a control: **no run on disk has ever
logged the mean-coordinate panel**, because it was added after every CIFAR run had
finished. Two of the supervisor's four requested deliverables are that curve.

## Task 2 — the generic-latent comparison arm (E15)

The question is whether the structured reverse-dynamics basis transfers better than
ordinary latent conditioning with the same transport map. The U-Net already conditions on
time through feature-wise linear modulation — every residual block reads a scale and
shift from `emb_proj(temb)` — so a generic latent enters the same path:

```python
# models/film_unet.py, registered as "film_unet"
temb = self.time_mlp(timestep_embedding(t, self.base_channels)) + self.z_mlp(z)

# baselines/film_conditioning.py -- a ScoreModel subclass with no basis term
class FiLMScoreModel(ScoreModel): ...

# runner/train_film.py -- uses the seam already in the baseline; do NOT edit training/loop.py
import models.film_unet
train(cfg, model_cls=FiLMScoreModel)
```

Everything downstream reaches the model only through `ScoreModel.eps_hat`, so the
substitution is a drop-in and the seven matching conditions in the protocol card hold
structurally rather than by bookkeeping. Keep centring **off** for this arm, so it aligns
with `ctr128_off`.

Report the parameter counts of both arms — `z_mlp` against the basis head, each as a
fraction of the backbone. `ALGORITHM.md` §9 warns that an equal step count is not an equal
budget when counts differ by orders of magnitude. Here they do not, so equal steps is
defensible; print the numbers rather than assert it.

Two outcomes are both worth having. Structured coordinates transporting better is the
paper's claim. The generic latent engaging where the low-dimensional basis does not says
the basis is the wrong parameterisation, and says it cleanly.

---

## Your files

Yours to modify: `models/score_model.py`, `training/loop.py`, `training/meta_train.py`,
`diagnostics/controls.py`, `config/base_config.py`, `runner/train.py`,
`tests/test_score_identity.py`.
Yours to create: `models/film_unet.py`, `baselines/`, `runner/train_film.py`.

**Do not touch** `scripts/`, `evaluation/`, `episodes/`, `adaptation/`, or any file under
`artifacts/`. Two other people own those.

## How to know you are finished

- The whole test suite passes with the centring flag both on and off.
- `|z_center|` does not diverge and coordinate spread has not collapsed below 0.05.
- `delta_task` has a mean and a 95 % interval over the final fifth of training, and the
  slope over the last 10 000 steps is reported.
- `check_backbone(FiLMUNet())` passes and `tests/test_backbone_swap.py` still passes.
- Both arms print their parameter counts.

## Deliver into `handoff/Jing-Peng/results/`

`E13_panel.txt`, `E15_film_panel.txt`, and — this one is easy to forget — **copies of the
`checkpoints/*_log.jsonl` files**, which are the primary evidence and are ignored by git
where they are written. They are a few hundred kilobytes each.

Checkpoints themselves are 430 MB and must not be committed. Keep them on the server; say
in your results file where they are.

## Finishing

```bash
cp checkpoints/ctr*_log.jsonl checkpoints/film128_log.jsonl handoff/Jing-Peng/results/
git add models training diagnostics config runner tests baselines handoff/Jing-Peng/results
git commit -m "E13 centred coordinates and E15 comparison arm"
git push origin jing-peng
```

Push to `jing-peng` and nowhere else. Do not merge into `main`, do not rebase onto another
package's branch, and check `git status` for a stray `.pt` before committing — a
checkpoint that slips past the ignore rule is painful to remove from history.

**Do not write a conclusion.** Deliver the numbers.
