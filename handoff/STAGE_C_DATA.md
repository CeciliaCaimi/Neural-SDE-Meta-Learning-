# Stage C data: what exists, what it supports, and what it does not

Written 2026-09-17 for a session that will run Stage C. Every number here was measured, not
projected; the commands that produced them are at the bottom. Read this before writing code
against Fitzpatrick17k — three of the facts below change the experimental design, not just
its scale.

---

## 1. Where the data is

```
FITZPATRICK_ROOT = C:\T1D Meta Learning\dataset\fitzpatrick17k
    images/              3887 files, 255 MB, named <md5hash>.<ext>
    fitzpatrick17k.csv   the annotation table, 16577 rows, 114 conditions
    manifest.csv         one row per attempted fetch: md5hash, label, fitzpatrick_scale,
                         host, url, status, bytes, path
```

**Outside the repository, and it stays outside.** The directory holding it also holds
clinical data; do not add anything above the repository root and do not commit images.
Reach the data through the `FITZPATRICK_ROOT` environment variable, as `domains/cifar100.py`
does for `CIFAR100_ROOT`. The three scripts below already default to the path above.

## 2. What was retrieved, and what cannot be

| Host | Rows | Share | Outcome |
|---|---|---|---|
| `atlasdermatologico.com.br` | 3905 | 23.6 % | **3887 retrieved, 99.5 %**; the 18 losses are 404 |
| `www.dermaamin.com` | 12631 | 76.2 % | **0 retrieved** |
| (no url) | 41 | 0.2 % | unretrievable by any route |

DermaAmin does not resolve. Neither `dermaamin.com` nor `www.dermaamin.com` has a working
delegation, for public resolvers as well as locally, so every request fails before it is
sent. This was checked on 2026-09-16 and again on 2026-09-17 with the same result. No user
agent, scheme or port changes it. **Three quarters of the dataset is unreachable and no
amount of retrying will change that**; only the authors' image release will, and that
request must be submitted by the project owner.

Counting only rows with a valid phototype (1–6; 565 rows carry −1):

| Population | Images | Conditions |
|---|---|---|
| Full dataset, if released | 16012 | 114 |
| Reachable archive | 3752 | 92 |
| **On disk now** | **3734** | **90** |

## 3. Three facts that change the design

**3.1 There is only one relation.** Phototypes I–III against IV–VI gives a single
source-to-target relation, where CIFAR had three transformations. In
`training/loop.py:144`, `n_relations` is set to `None` when the corruption count is 1, so
the relation descriptor `c` is omitted entirely and `transport.relation_emb` is `None`.
Consequences: the `relation_only` control degenerates to `Delta_gamma(0)`, a single
constant, and becomes nearly indistinguishable from the mean-source control. **The A1
control panel has to be redesigned for this setting; it cannot be copied across.**

**3.2 `transform_signature` does not transfer.** The transformation statistic used
throughout Stage A is three fixed convolutions built to read Gaussian blur, additive noise
and contrast reduction. It says nothing about a phototype shift. **C2's frozen evaluation
predictors are not optional here — they are the instrument the entire result rests on**, and
they must report their own held-out ceiling beside every verdict they are used for.

**3.3 The source side binds, not the target side.** An earlier count called a condition
usable when its target side could fund a 20-image support reserve plus a 25-image held-out
query, and reported 8. That omits the source side. A condition also has to fund `M_S` source
images, and a *training* task a source query batch on top, because `src_query` feeds the
source denoising term of the meta objective. Nothing at meta-test reads `src_query`:
evaluation touches `src_support`, `tgt_support` and `tgt_query` only.

| `M_S` | Train-capable | Eval-capable |
|---|---|---|
| 64 (what Stage A trained with) | **2** | 6 |
| 32 | 6 | 7 |
| **16** | **6** | **12** |
| 8 | 7 | 13 |

**Use `M_S = 16`.** A4 measured that source evidence saturates below sixteen images — across
a nineteenfold sweep every comparison moved less than its own interval half-width — so the
reduction costs nothing and triples the usable conditions. A4 is what makes Stage C
constructible on the reachable archive at all.

## 4. The split this forces

At `M_S = 16`, source query batch 32, support reserve 20, held-out query at least 25.
Only six conditions can fund a source query batch, so **the training set is forced, not
chosen**.

| Split | Tasks | Source needed per task | Reserve | Held-out target | Target total |
|---|---|---|---|---|---|
| train | 6 | 48 | 120 | 352 | **472** |
| val | 3 | 16 | 60 | 208 | **268** |
| test | 3 | 16 | 60 | 122 | **182** |
| total | 12 | — | 240 | 682 | **922** |

Source images consumed: 288 / 48 / 48, total 384. Of the 1668 images these twelve conditions
hold, 1306 are read; the remainder is source stock beyond `M_S` that no stream touches.

```
train  lichen planus            N_S= 73  N_T=152     porokeratosis actinic   N_S= 65  N_T= 59
       basal cell carcinoma     N_S=169  N_T= 84     nematode infection      N_S= 67  N_T= 58
       squamous cell carcinoma  N_S=133  N_T= 63     scabies                 N_S= 70  N_T= 56
val    pityriasis rubra pilaris N_S= 26  N_T=146     psoriasis               N_S= 29  N_T= 69
       pityriasis rosea         N_S= 42  N_T= 53
test   lupus erythematosus      N_S= 30  N_T= 77     dariers disease         N_S= 22  N_T= 51
       papilomatosis conf. ret. N_S= 20  N_T= 54
```

`N_S` and `N_T` are the images a condition *has* on the source and target side. `M_S` and
`K_T` are how many are *used*. Do not confuse the pair.

## 5. Choosing `K_T`

Only the **maximum** costs data: the support reserve is a nested prefix
(`tgt_support_reserve[:k]`), so extra values of `K_T` cost compute and no images.

| max `K_T` | Conditions with held-out query ≥ 25 | ≥ 20 | ≥ 15 |
|---|---|---|---|
| 10 | **15** | 16 | 19 |
| 20 | 12 | 13 | 15 |
| 30 | 9 | 12 | 12 |
| 40 | 5 | 6 | 9 |

Two defensible choices:

- **`max K_T = 20`, e.g. {1, 2, 3, 5, 8, 12, 20}** — matches Stage A, keeps 12 conditions.
- **`max K_T = 10`, e.g. {1, 2, 3, 5, 8, 10}** — 15 conditions, one more held-out test task.

A1 found the method's margin over target-only modelling falls from `+0.0167` at one target
image to `+0.0044` at twenty, and the whole decay happens over five sampled points. A denser
grid at the small end is where the information is. On three test tasks the `K_T = 20` cell
cannot resolve `+0.0044` anyway, which argues for the second choice.

## 6. What this supports, and what it does not

**Achievable.** C1, C2 and C3 in full — they need images, not a wide task split. An
A1-style source-dependence panel on real clinical data, which is the strongest result
available at this scale. A scarcity curve reported as a directional claim.

**Not achievable.** C4's headline as the plan specifies it: a target-data equivalence
statement of the form *the method at `K_T = 2` matches target-only at `K_T = 10`*. That needs
a paired interval over held-out tasks, and there are three.

**Why three is the binding number.** The unit of replication is the *task*, not the image. A
task is (condition, relation); with one relation, one condition is one task. CIFAR evaluated
20 classes × 3 transformations = **60 episodes per `K_T`**; here the test side gives
**3**. Resampling the source support set produces more episodes but not more tasks, and
between-task variance is the term that dominates — the repository's own `CLAUDE.md` records
it at roughly sixteen times the effect size. More images, more samples per cell and a denser
`K_T` grid all leave the three untouched.

A second instrument limit, independent of `K_T`: the test tasks hold **57, 34 and 31**
held-out target images. Stage A evaluated against a query batch of 128. A sliced Wasserstein
distance computed against 31 real images is materially noisier than anything in Stage A, and
any comparison across the two stages should say so.

**Two ways to move the three**, and only two:

1. **Leave-one-condition-out cross-validation** — train twelve models, each holding out one
   condition, giving twelve held-out readings from models that never saw the held-out task.
   At roughly 70 minutes per training run (the A5 variants' rate) that is about 14 hours. It
   departs from the plan's train/validation/test task split but still makes the relation the
   object of generalisation, which is what that split was for; say so in the write-up.
2. **The authors' image release** — 39 conditions clear C4's target-side threshold on the
   full dataset against 8 now. This is blocking, slow, and outside the repository's control.

## 7. What Stage A predicts about the outcome

- **A3**: the margin over reusing the source coordinate was the largest and most robust
  effect of the stage. Expect it to reproduce.
- **A1**: the margin over target-only modelling decayed fourfold from `K_T = 1` to
  `K_T = 20`, because a single target image already revealed the task. A phototype shift
  changes colour statistics markedly, so the denoising objective will feel it — but that
  same fact means it is likely readable from one target image too. **Expect the same decay.**
  The demographic experiment inherited dSprites' role of supplying a task variable that
  cannot be read from a single target example; on A1's evidence it may not fill it.
- **A4**: `M_S = 16`, as above.
- **A5**: use the linear transport map, not the residual multilayer perceptron. They were
  indistinguishable at all three budgets on CIFAR with nineteen times fewer parameters, and
  352 parameters will behave better across six training tasks than 6824.

## 8. Reproducing and extending

```bash
python scripts/fetch_fitz.py
```

Resumable; files already present are skipped, so re-running costs nothing. Pass
`--host www.dermaamin.com` to re-confirm the outage, `--limit N` to sample.

```bash
python scripts/check_fitz.py
```

Counts what is on disk against the plan's thresholds.

```bash
python scripts/split_fitz.py
```

The `M_S` table of §3.3 and the allocation of §4.

Full records: `handoff/results/C0_feasibility.txt` (the original gate),
`C0b_retrieval.txt` (what the full fetch returned), `C0c_split.txt` (the source-side
correction and the split). Where C0 and C0b disagree with C0c, **C0c is right**: the earlier
two counted only the target side.
