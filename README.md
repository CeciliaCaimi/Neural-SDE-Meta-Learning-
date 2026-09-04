# Meta-Diffusion

Few-shot adaptation of a diffusion model by optimising **a handful of numbers** instead of
retraining the network.

A diffusion model normally needs thousands of images to learn a new distribution. Here the
network is trained once across many tasks and then **frozen**. Adapting it to a new target
distribution means fitting a `k`-dimensional vector (`k = 16` by default) — sixteen scalars
against a network of 27.8 million weights.

The other half of the idea: when only one or two target images exist, even sixteen numbers are
hard to fit. So a small **transport network** predicts the target coordinate from the abundant
*source* data alone, without ever seeing a target image, and gives the fit a starting point.

---

## The one-paragraph version of the method

A shared backbone maps a noised image to features. Two heads read those features: a **base
head** producing the ordinary noise prediction, and a **basis head** producing `k` directions
along which that prediction can be perturbed. The task coordinate `z` weights those directions:

```
eps_hat(x_t, t, z) = eps_hat_0(x_t, t) + sum_l z_l * R_l(x_t, t)
```

Everything on the right except `z` is frozen at deployment. Whether the method works reduces to
one question: **can those `k` learned directions span the differences that actually separate one
task from another?**

`ALGORITHM.md` states this properly, with every equation, hyperparameter and diagnostic needed
to reimplement the project from scratch.

---

## Status, honestly

| Stage | Domain | Result |
|---|---|---|
| 1 | 2-D Gaussian mixtures (analytic ground truth) | **Works.** Transport beats every baseline at small `K_T`, with paired confidence intervals over 32 tasks x 3 seeds |
| 2 | dSprites | Not started |
| 3 | CIFAR-100 | **Partly works, and the failure is localised.** The encoder and the transport map both do their job; the learned basis cannot turn a correct coordinate into a denoising gain |

The CIFAR-100 outcome is a real finding, not a bug, and it is documented rather than hidden.
It is also more specific than "it does not work": of the method's three moving parts, two are
measurably correct on natural images and one is not. See
[Findings on CIFAR-100](#findings-on-cifar-100) below.

---

## Quick start

Requires Python 3.11+, PyTorch with CUDA, and roughly 2 GB of GPU memory.

```bash
pip install torch numpy
```

CIFAR-100 is expected at `<repo>/../../../dataset/cifar-100-python/` (the standard
`cifar-100-python` directory from the official `.tar.gz`). Adjust the path in
`domains/cifar100.py` if yours lives elsewhere.

**Verify the installation** — 72 assertions, no GPU needed for most:

```bash
python -m tests.test_score_identity
python -m tests.test_backbone_swap
python -m tests.test_gmm_analytic
python -m tests.test_cifar100_episodes
python -m tests.test_document_conformance
```

**Stage 1** — the analytic validation, about 20 minutes on a modern GPU:

```bash
python -m runner.stage1_gmm --k 16 --steps 80000
```

**Build the CIFAR-100 splits**, then train:

```bash
python -m runner.build_cifar100_splits
python -m runner.train --scheme domainshift --steps 50000
```

**Evaluate** a checkpoint against every baseline, with paired confidence intervals:

```bash
python scripts/cifar_table.py checkpoints/<run>_step50000.pt
```

---

## What is where

```
models/          the network
  backbone.py      THE swap seam: implement this to use a different diffusion network
  score_model.py   the meta layer: base head + basis head + coordinate fusion
  set_encoder.py   r_psi, permutation invariant; source and target share it
  transport.py     Delta_gamma, the reusable source-to-target map
  unet.py          reference backbone (a small DDPM U-Net)
  mlp_backbone.py  vector backbone for stage 1

episodes/        the data protocol
  splits.py        sibling-class scheme (a hard, low-relatedness stress test)
  domainshift.py   clean-to-corrupted scheme (the main experiment)
  guards.py        leakage assertions that enforce the protocol rather than describe it

adaptation/      deployment-time coordinate refinement and every baseline strategy
diffusion/       schedule, forward noising, losses, samplers
diagnostics/     the stopping conditions, run automatically during training
training/        one meta-training step, and the loop around it
runner/          entry points
scripts/         one-off analysis and table generation
tests/           72 assertions, including conformance to the source specification
```

### Swapping the diffusion network

The whole point of `models/backbone.py` is that this costs one file:

```python
@register_backbone("my_dit")
class MyDiT(DiffusionBackbone):
    @property
    def spec(self): return BackboneSpec(feature_channels=256, image_channels=3, image_size=32)
    def forward_features(self, x_t, t): ...   # -> (B, 256, 32, 32)
```

Then `check_backbone(MyDiT())` validates the contract, and `--backbone my_dit` uses it.
`tests/test_backbone_swap.py` demonstrates this with a token-based transformer and with a
backbone that supplies its own noise prediction; the meta layer is not touched in either case.

---

## The experimental protocol

Three rules keep the results honest, and the code enforces all three:

**Semantic tasks are split, not images.** The 20 CIFAR-100 superclasses divide 12 / 4 / 4 into
train / validation / test, giving 60 / 20 / 20 fine classes. A class belongs entirely to one
split. This tests whether the *relation* generalises, rather than whether particular task pairs
were memorised.

**Hyperparameters are chosen on validation only.** Test classes are touched once, for the final
number. Diagnostics during training read the validation split.

**Target query data never reaches refinement.** This is enforced at the type level:
`adaptation.coordinate.refine()` has no parameter through which it could arrive.

Each episode carries four mutually disjoint streams:

| Stream | Size | Purpose |
|---|---|---|
| source support | 300 (M_S) | infer the source coordinate `z_S` |
| source query | 50 | the source denoising loss |
| target support | `K_T` in {1, 2, 5, 10, 20} | the only target evidence available at deployment |
| target query | 230 | target and transport losses in training; evaluation only at test |

---

## Findings on CIFAR-100

Three variants were run to convergence. Each fixed the specific problem it targeted, and the
downstream result did not change.

**Sibling fine classes as source and target.** The coordinate collapses: its relative spread
across episodes stays near 0.02 against a 0.05 threshold. Dissecting a checkpoint showed the
pooled encoder features barely separate one flower species from another, so the encoder cannot
extract task identity from the denoising gradient alone.

**One fixed corruption (clean to blurred).** Collapse is solved — coordinate spread rises to
0.32 and source is cleanly separated from target. But the benefit vanishes: `gain_vs_zero`
peaks at 0.084 around step 2000 and decays to 0.0002 by step 50000, while basis usage falls
from 0.41 to 0.012. As the 27.8M backbone matures it models the union of clean and blurred
images unconditionally, leaving the coordinate nothing to do.

**Three corruptions (blur, noise, contrast).** The hypothesis was that a backbone unable to
know *which* corruption applies would be forced to use the coordinate. It was not. The same arc
repeats: gain peaks at 0.078 and decays to 0.0016, basis usage from 0.29 to 0.039. The encoder
meanwhile keeps improving, reaching a spread of 1.21.

Paired evaluation over 40 held-out episodes puts numbers on it:

| Strategy | K_T=1 | K_T=5 | K_T=20 |
|---|---|---|---|
| no adaptation (z=0) | 0.0902 | 0.0902 | 0.0902 |
| target support only | 0.0888 | 0.0887 | 0.0887 |
| reuse z_S directly | 0.0901 | 0.0890 | 0.0888 |
| transport, no refine | 0.0887 | 0.0887 | 0.0887 |
| **transport + refine** | **0.0887** | **0.0887** | **0.0887** |
| oracle (fit on target) | 0.0887 | 0.0887 | 0.0887 |

Any nonzero coordinate beats `z=0` by 0.0015 (95% CI ±0.0002) — statistically real, but 1.7% of
the loss. No coordinate beats any other: transport, refinement and even the oracle upper bound
all land on the same number, and `K_T` changes nothing. The model learned to use the basis as a
small fixed offset, not as a task-specific direction.

### Where exactly it fails

The table above says the coordinate does not help. It does not say *which* part is at fault, and
the obvious guesses — the encoder cannot read a task, or transport cannot predict one — both turn
out to be wrong.

Measuring the coordinates directly against a reference encoded from 230 target images
(`scripts/probe_transport.py`, 40 validation episodes, single-corruption checkpoint):

| K_T | source coordinate | after transport | gap closed | K_T-sample encoding |
|---|---|---|---|---|
| 1 | 3.106 ±0.073 | **0.227 ±0.019** | +2.879 ±0.070 | 0.397 ±0.047 |
| 5 | 3.106 ±0.073 | **0.227 ±0.019** | +2.879 ±0.070 | 0.180 ±0.029 |
| 20 | 3.106 ±0.073 | **0.227 ±0.019** | +2.879 ±0.070 | 0.112 ±0.009 |

Transport closes **93% of the distance** to the true target coordinate, with a confidence
interval nowhere near zero. At `K_T = 1` the transported coordinate (0.227) is *closer to the
truth than the coordinate encoded from the single available target image* (0.397) — which is
precisely the claim the method makes. Abundant evidence overtakes it between five and twenty
samples, exactly where it should.

The first three columns do not vary with `K_T` by construction, since `z_S` and `z_tilde_T` are
both built from source data alone and the third column is their difference. Only the last column
depends on `K_T`, and it falls monotonically, as a sparse estimate should.

So the encoder reads tasks, and transport predicts them. What fails is the last step: the
coordinate is multiplied into the learned basis `R_1..R_k` and nothing happens — basis usage
sits at 0.012 to 0.039, and a correct coordinate denoises no better than a wrong one.

**The bottleneck is the basis, not the coordinate.** On natural images a backbone of this
capacity models the domains jointly, leaving the basis with no residual structure to represent;
it degenerates into a small fixed offset. The source document raises this possibility directly.
Whether it follows from natural images or merely from *this backbone size* is not settled —
shrinking the backbone (`--base-channels`) is the experiment that separates the two, and it has
not been run.

Stage 1, where the backbone is small and the tasks genuinely diverse, shows all three parts
working together. That contrast is the most informative result so far.

---

## Reproducing the analysis

```bash
# stage 1, all baselines with paired confidence intervals
python -m runner.stage1_gmm --k 16 --steps 80000
python scripts/paired_ci.py
python scripts/final_table.py

# CIFAR: where a coordinate collapse happens, layer by layer
python scripts/probe_collapse.py checkpoints/<run>_step5000.pt

# CIFAR: does transport move z_S toward an abundant-target reference?
# This is what separates "the coordinate is wrong" from "the basis cannot use it".
python scripts/probe_transport.py checkpoints/<run>_step50000.pt --split val
```

Training writes one JSON line per logging step to `checkpoints/<run>_log.jsonl`, and prints a
diagnostic block on the validation split at a configurable interval. Two numbers matter most:

- `gain_vs_zero` — does the correct coordinate beat `z = 0`?
- `r_basis` — is the basis contributing anything at all?

When either approaches zero, the framework says so and names the failure mode. That is
deliberate: the guardrails are stopping conditions, not a post-mortem.
