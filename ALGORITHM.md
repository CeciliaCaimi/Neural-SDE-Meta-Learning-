# Meta-Diffusion: a complete specification

This document is written so that someone — or an AI coding assistant — can reimplement the
method from scratch without reading the source. Every equation, dimension, hyperparameter and
diagnostic that the working implementation depends on is stated here, together with the
mistakes that are easy to make and the checks that catch them.

Notation follows the paper the project is based on; equation numbers in parentheses are that
paper's.

---

## 1. The problem

A diffusion model of a target distribution normally needs thousands of samples. We want one
that adapts to a new target from `K_T` samples, where `K_T` is between 1 and 20, while a large
*source* distribution related to it has `M_S` samples with

```
M_S >> K_T                                                                            (16)
```

This asymmetry is the defining condition. A "task" `y` is a pair of related distributions
`(p_{y,S}, p_{y,T})`; the S-to-T relation recurs across tasks and is what makes transfer
possible at all.

## 2. The central idea

Train one diffusion backbone across many tasks, then **freeze everything** and let each new
task be described by a low-dimensional coordinate `z` in R^k.

Score functions and noise predictions are interchangeable under a known time scale:

```
eps*(x_t, t) = -sigma_t * s*(x_t, t)                                                  (19)
```

so the model is implemented in `eps` space, compatible with any standard DDPM code, while the
claim is naturally stated in score space. The claim is that within a related family of tasks,
the *difference* between task score fields is low-dimensional — not that any single
distribution is.

The parameterisation:

```
eps_hat_{phi,z}(x_t, t) = eps_hat_{phi,0}(x_t, t) + sum_{l=1..k} z_l * R_{phi,l}(x_t, t)   (21)
```

`eps_hat_0` is the shared baseline prediction; `R_1..R_k` are learned perturbation directions;
`z` weights them. At deployment `phi` is frozen and only the `k` scalars of `z` move.

**The method succeeds or fails on one question:** can `R_1..R_k` span the differences that
actually separate tasks? Sections 8 and 9 give the diagnostics that answer it directly, and
they must be run — a plausible-looking sample is not evidence.

## 3. Architecture

Four modules. Everything high-dimensional is shared; everything task-specific is compressed
into `z`.

### 3.1 Shared backbone

Maps `(x_t, t)` to a feature tensor `H` of shape `(B, C_feat, H, W)` at input resolution.

**It must not contain a noise-prediction head.** Both heads belong to the layer above. This is
what makes "one backbone, not `k` separate denoisers" true, and it is what makes the backbone
swappable.

Reference implementation: a small DDPM U-Net, base width 128, channel multipliers (1, 2, 2),
2 residual blocks per level, attention at 16x16. 27.74M parameters. Architecture is a
validation choice, not a contribution.

The contract a replacement must satisfy:

- output is a spatial feature map at input resolution (token models unpatchify internally)
- genuinely depends on `t` — the same `x` at different `t` gives different features
- deterministic in eval mode
- if it already has an `eps_hat` head (e.g. a pretrained DDPM), expose it and declare
  `provides_eps=True`; the layer above will reuse it and train only the basis head

A subtlety worth stating: many public implementations zero-initialise the output convolution of
every residual branch. Such a network is exactly the identity **at initialisation**, so a naive
`t`-sensitivity test fails on a perfectly sound architecture. Perturb the parameters and retry
before declaring failure.

### 3.2 Two prediction heads

Both read the same `H` from **one** backbone pass.

- **base head**: `C_feat -> C_img` convolution giving `eps_hat_0`. 3,459 parameters.
- **basis head**: `C_feat -> k * C_img` convolution, reshaped to `(B, k, C_img, H, W)`.
  55,344 parameters at `k = 16`, `C_img = 3` (48 output channels).

Initialise the basis head small but **nonzero** (std about 1e-3). Zero initialisation pins
`dL/dz` at exactly 0, so the encoder receives no gradient through the basis path and never
starts learning.

### 3.3 Set encoder `r_psi`

Maps a set of samples to a coordinate, permutation invariantly:

```
r_psi(D) = rho_psi( (1/m) * sum_i h_psi(x_i) )                                        (22)
```

`h_psi` for images: three stride-2 convolution blocks 32 -> 16 -> 8 -> 4 with GroupNorm and
SiLU, global average pool, linear to `feature_dim = 256`. `rho_psi`: `256 -> 256 -> k`.
0.33M parameters total.

**Source and target share this one encoder.** There is no separate source-specific or
target-specific network.

Mean pooling is a validation choice. If diagnostics show `z` failing to separate the two
domains, change the pooling — concatenating the per-dimension standard deviation is the obvious
first move — before splitting the encoder. If you do add a standard deviation, compute it as
`sqrt(var + eps)` with `eps` about 1e-5: a single-element support set (`K_T = 1`) has exactly
zero variance, and the gradient of `sqrt(0)` is infinite, which turns the model to NaN within a
few steps. The forward pass looks fine, which is what makes this one expensive to find.

Because mean pooling is linear, the source coordinate can be accumulated in chunks at
deployment, so `M_S` is not bounded by memory there. During training it is, since gradients
must be retained.

### 3.4 Transport `Delta_gamma`

```
z_tilde_{y,T} = z_{y,S} + Delta_gamma(z_{y,S}, c_{S->T})                              (25)
```

A residual MLP `(k + d_c) -> 64 -> 64 -> k`; 6,288 parameters at `k = 16` with `c` omitted.
Initialise the final layer small so that `z_tilde_T` is approximately `z_S` at the start —
equivalent to the "reuse the source coordinate" baseline — making any later gain attributable.

Two constraints that are easy to get wrong:

**The input contains no target sample.** Transport must predict the target coordinate having
never observed target data. This is the proposition under test; if a target image reaches it,
the experiment measures nothing.

**The relation descriptor `c` is indexed by relation, never by semantic class or by any
grouping that is disjoint between train and test.** Indexing by superclass, for instance, means
meta-testing reads embedding vectors that were never trained, silently disabling transport.
Omit `c` entirely for a single fixed relation.

## 4. Data protocol

An episode is the tuple

```
e_y = (y, d_S, d_T, D^s_{y,S}, D^q_{y,S}, D^s_{y,T}, D^q_{y,T}, c_{S->T})
```

with four **mutually disjoint** streams. Support sets compute coordinates; query sets compute
losses. Nothing else.

### 4.1 The main experiment: clean to corrupted

One semantic task = one fine class. The source domain is that class's clean images; the target
domain is *different* images of the same class under a fixed, deterministic, semantics-
preserving transformation. The transformation is the relation, learned on training classes and
tested on entirely held-out classes.

Using different underlying images for source and target matters: pairing `clean(X)` with
`corrupt(X)` leaks, and real domain shift does not come with paired samples.

Split by superclass so that the three ways are semantically apart, not merely disjoint by id:

| | superclasses | fine classes |
|---|---|---|
| train | 12 | 60 |
| validation | 4 | 20 |
| test | 4 | 20 |

Per class, from 600 images: 350 to the source pool (300 support + 50 query) and 250 to the
target pool (20 reserved for support + 230 query).

Corruptions must be reproducible from `(image id, type, severity)`. Blur and contrast are
deterministic functions; additive noise needs its seed derived from the image id, or the
"same" target domain differs between runs.

### 4.2 The stress test: sibling classes

A harder, less related variant in which source and target are two *different* fine classes of
one superclass. Since CIFAR-100 is perfectly balanced, the target's scarcity is imposed by
restricting its support set, not inherited from the data. Keep this as a low-relatedness
stress test; it is not the main experiment.

### 4.3 Rules the code should enforce, not merely document

- semantic tasks split three ways and disjoint; a class belongs entirely to one split
- support and query disjoint within each domain
- target support **nested** over `K_T`: the image used at `K_T=1` also appears at `K_T=2`, so
  the `K_T` sweep is a within-group comparison rather than a lottery over which images appeared
- every hyperparameter and adaptation budget selected on validation classes, then frozen
- target query never enters refinement — best enforced by giving the refinement function no
  parameter through which it could arrive

## 5. Meta-training

Per step, one episode:

```
z_S        = r_psi(D^s_{y,S})                       source support -> coordinate
z^enc_T    = r_psi(D^s_{y,T})                       target support -> coordinate (baseline branch)
z_tilde_T  = z_S + Delta_gamma(z_S, c)              transport; no target sample involved
```

Three denoising losses, all of the form `w(t) * ||eps - eps_hat||^2`:

```
L_src   = E over D^q_{y,S} of loss with z_S                                           (27)
L_tgt   = E over D^q_{y,T} of loss with z^enc_T                                       (28)
L_trans = E over D^q_{y,T} of loss with z_tilde_T                                     (29)

L_meta  = L_src + lambda_T * L_tgt + lambda_tr * L_trans + lambda_z * L_z             (30)
L_z     = ||z_S||^2 + ||z^enc_T||^2
```

`L_tgt` and `L_trans` run on the **same** target query batch. Give them the same noise draw and
the same backbone pass as well: it is a paired comparison, so the variance of their difference
falls sharply, and it costs one forward pass instead of two.

`z_tilde_T` is excluded from `L_z`. Regularising it would penalise transport for moving.

**Do not differentiate through the refinement inner loop.** `Delta_gamma` is trained by
`L_trans` directly; refinement happens only at meta-test.

Defaults: 20k–50k steps, AdamW at 2e-4 with 500 warmup steps, gradient clip 1.0, EMA decay
0.999, bf16 autocast, `lambda_T = lambda_tr = 1`, `lambda_z = 1e-4`, `M_S = 64` images per step
(subsampled from the 300-image pool for memory), query batch 32, cosine schedule with 1000
steps, uniform `w(t) = 1`.

Evaluate and diagnose under the EMA weights. Otherwise the reported numbers come from different
parameters than the checkpoint.

## 6. Deployment

Everything freezes. `z` is the only variable:

```
z*_T = argmin_z { (1/K_T) * sum_{i=1..K_T} E_{t,eps} [ w(t) ||eps - eps_hat_{phi,z}(x^T_{i,t}, t)||^2 ]
                  + (beta_0 / K_T) * ||z - z_tilde_T||^2 }                            (26)
```

Both divisions by `K_T` earn their place. Dividing the denoising term keeps the gradient scale
stable across support sizes. Dividing the prior term by the same quantity makes the
source-derived prior's relative influence decay as target evidence accumulates: it dominates at
one sample and recedes once samples are plentiful.

Defaults: `J = 25` steps, Adam at `eta_z = 1e-2`, `beta_0 = 1`, 16 `(t, eps)` draws per step.

### 6.1 Baselines, all sharing one budget object

| Strategy | Initial value | Prior centre |
|---|---|---|
| no adaptation | `z = 0`, no refinement | — |
| target only | `z^enc_T` | itself |
| reuse source | `z_S` | itself |
| transport, no refine | `z_tilde_T`, `J = 0` | — |
| **transport + refine** | `z_tilde_T` | `z_tilde_T` |
| oracle | refined on abundant target data | none |
| full fine-tuning | all weights, `z = 0` | — |

Sharing one budget object makes the matched comparison structural rather than a thing someone
has to remember.

Two traps here. First, "no adaptation" must mean `z = 0` **and no refinement**; skipping
refinement only when `J = 0` silently turns it into "refine from zero", which is a different
method. Second, full fine-tuning at a matched `J` is the correct *fair baseline* but a useless
*upper bound*: a learning rate suited to 16 scalars barely moves a million weights. Report both
the matched-budget number and the converged one, and take the converged one as the minimum over
a budget grid — full fine-tuning overfits a small target set non-monotonically.

## 7. Sweeps

- coordinate dimension `k` in {16, 32, 64}, chosen on validation; take the smallest that is not
  clearly capacity-limited
- target support `K_T` in {1, 2, 5, 10, 20}
- source evidence `M_S` in {16, 32, 64, 128, 256, all}, at small `K_T`
- transformation severity from mild to severe, to locate the transition from positive to zero
  or negative transfer

## 8. Diagnostics — the part that is not optional

Run these on held-out validation episodes *during* training. They are stopping conditions; read
after the fact they are only an autopsy.

**Basis usage.** `r_basis = ||sum_l z_l R_l|| / (||eps_hat_0|| + eps)`, per sample. Near zero
means the basis is unused and the model is a plain unconditional diffusion model wearing a
coordinate.

**The three-way control.** Compare the denoising loss under the correct `z`, under `z = 0`, and
under a **shuffled** `z`. If `z = 0` matches the correct coordinate, no task-specific structure
has been established.

One easy mistake: every query image in an episode shares one `z`, so shuffling along the batch
dimension does nothing at all. A real shuffled control uses **another episode's** coordinate.

**Coordinate spread.** Mean pairwise distance between the coordinates of different episodes,
divided by mean norm. Below about 0.05 the encoder is emitting a near-constant, and `z = 0`
comparisons stop meaning what they appear to mean: the basis is then acting as extra shared
capacity, not as task-specific structure. Check this *before* interpreting the three-way
control, since collapse forces the shuffled comparison to zero for an unrelated reason.

**Source-to-target separation.** `||z_S - z^enc_T||` relative to `||z_S||`. If the encoder does
not separate the two domains, nothing downstream can.

**Transport displacement.** `||z_tilde_T - z_S||`. Constantly zero means `Delta_gamma` has
degenerated to the identity and the method has become "reuse the source coordinate".

**Distance to an abundant-target reference.** Encode a *large* target sample to get
`z_T^abund`, then compare `||z_S - z_T^abund||` before transport with
`||z_tilde_T - z_T^abund||` after. This separates two questions that are otherwise conflated:
whether transport learned the relation, and whether the basis can turn that into a denoising
gain. Comparing against `z^enc_T` cannot do this, because at `K_T = 1` the reference is itself
noise.

Report `||z^enc_T - z_T^abund||` in the same table. It is the only one of the three that varies
with `K_T`, and watching it fall as `K_T` grows tells you where sparse encoding overtakes
transport — the crossover point is the regime boundary the method claims to own. If this
diagnostic shows transport working while the denoising gain stays at zero, the fault is in the
basis, and no amount of work on the encoder or the transport map will help.

## 9. Evaluation

Where ground truth is analytic, report relative score-field error at points drawn from the true
noised distribution — the region the sampler actually visits, not a uniform grid. On images,
where no ground truth exists, use held-out target denoising loss, plus distributional distances
such as sliced Wasserstein or energy MMD on generated samples.

**Pair everything.** Run all strategies on the same episode with the same noise draw, take the
per-episode difference, then average. Between-task variance is typically an order of magnitude
larger than the effect, so an unpaired comparison over a handful of tasks returns an arbitrary
sign. Report confidence intervals across held-out tasks and seeds, and choose the number of
tasks from the effect size you expect rather than from convenience.

Two more habits worth adopting. Read any "this method attains X" claim only where the curve
against training steps has levelled; the same quantity can differ by twenty percentage points
when training is short. And when a comparison method differs by orders of magnitude in
parameter count, an equal step count is not an equal budget.

## 10. Sampling

All methods must share one sampler, or sample quality is not comparable. The sampler needs only
a callable `eps_fn(x_t, t)`, so it stays ignorant of both backbone and coordinate mechanism.

Clip the predicted `x_0`. This is not cosmetic: under a cosine schedule `alpha_bar` at the last
step is about 2.4e-9, so

```
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
```

amplifies any error in `eps_hat` by roughly 2e4, and the trajectory diverges within a few steps
at high noise. Use the posterior mean of `q(x_{t-1} | x_t, x0_hat)` so the clip has somewhere to
apply. Images clip to [-1, 1]; other domains need a bound derived from the data.

## 11. Recommended order of work

1. **A domain with analytic ground truth**, such as 2-D Gaussian mixtures. Gaussian noising
   preserves the mixture form, so the true score has a closed form and model output can be
   compared against truth directly rather than through sample appearance. Failing here is
   orders of magnitude cheaper than failing on images, and cannot be blamed on an untrained
   backbone. This is the first go/no-go.
2. **A controlled image domain with known generative factors**, such as dSprites, where factor
   recovery can be measured directly.
3. **Natural images**, only once the mechanism is known to engage.

Verify the analytic score itself by at least two independent routes — autograd through the
log-density and finite differences — before trusting anything built on it.

## 12. Failure modes, and what each one means

| Symptom | Reading |
|---|---|
| `r_basis` near 0 | the basis is unused; the model is unconditional |
| `z = 0` as good as the correct `z` | no task-specific reverse-dynamics structure was established |
| coordinate spread near 0 | the encoder emits a near-constant; fix this before reading other controls |
| `z_S` and `z^enc_T` not separated | the encoder cannot tell the domains apart; change pooling before splitting the encoder |
| transport displacement near 0 | `Delta_gamma` has become the identity |
| gain peaks early then decays to 0 | the shared backbone is absorbing the task difference; the coordinate has no work left |

That last row is worth dwelling on, because it is what this project observed on natural images,
and because the obvious diagnosis was wrong.

The encoder kept improving — coordinate spread rose steadily — while basis usage fell and the
advantage over `z = 0` decayed to nothing. The natural conclusion is that the coordinate had
become meaningless. Measuring it against an abundant-target reference says otherwise: transport
closed 93% of the distance from the source coordinate to the true target coordinate
(3.106 to 0.227, CI ±0.07 over 40 validation episodes), and at `K_T = 1` the transported
coordinate was closer to the truth than the coordinate encoded from the one available target
image. The coordinate was right. Multiplying it into the basis simply produced nothing.

This is why the abundant-reference diagnostic earns its place. Without it, "no gain" is a single
undifferentiated failure; with it, the pipeline splits into three testable stages — can the
encoder read a task, can transport predict one, can the basis express it — and the failure
localises to the third. Those three call for entirely different remedies, so collapsing them
into one number wastes the experiment.

Whether the remedy is a smaller backbone, a more diverse task family, or accepting that the
hypothesis does not hold in this regime is open. The honest thing is to report the curves and
the stage-by-stage diagnostic rather than the endpoint alone.
