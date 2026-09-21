# Meta-Diffusion final protocol lock

Locked on 2026-09-21 before inspecting any new pathology-family Meta-Diffusion output.
Repository HEAD at lock: `6de5187426b4ebfb4545846a49ca4d30f02987cb` on branch
`jinjian`. The working tree contains pre-existing uncommitted experiment work; new final-plan
artifacts are isolated under `experiments/meta_diffusion_final/`.

## Frozen method

- Fitzpatrick model recipe: 32x32 small U-Net, base width 32, additive basis, coordinate
  dimension 16, source budget 32, linear transport, 20,000 training steps, AdamW at 2e-4,
  EMA 0.999, seed 12345.
- Headline inference uses linear transport with no target-time refinement. No new head,
  larger coordinate, nonlinear transport, or architecture change is permitted.
- Population relation: Fitzpatrick I--III (source) to IV--VI (target).
- Evaluation budgets: K_T in {0,1,2,5,10,20}. Target-only is N/A at K_T=0.
- Generation: DDIM eta=0, 50 steps, 96 samples per cell, evaluation seed 4321.

## P1: pathology-family semantic resolution

The task field is the official `nine_partition_label`. Eligibility is determined without
model output: a family must have at least 64 clean, deduplicated source images and 45 clean,
deduplicated target images. Sixty-four source images fund a 32-image training source query
and M_S=32; 45 target images fund a nested 20-image support reserve and at least 25 disjoint
query images.

Six families meet the pre-model count rule. Because six tasks cannot support a reliable
fixed train/validation/test partition, use three locked task-level folds. Families are sorted
alphabetically, paired consecutively as test tasks, the next eligible family cyclically is
validation, and the remaining three are training tasks:

| Fold | Train families | Validation family | Test families |
|---|---|---|---|
| 0 | inflammatory; malignant epidermal; malignant melanoma | genodermatoses | benign dermal; benign epidermal |
| 1 | benign dermal; benign epidermal; malignant melanoma | malignant epidermal | genodermatoses; inflammatory |
| 2 | benign epidermal; genodermatoses; inflammatory | benign dermal | malignant epidermal; malignant melanoma |

Within each fold, the wrong-family source for either test family is the other test family.
This rule is deterministic and does not use outcome values. Every family is an outer-test
task exactly once. Checkpoints are selected using minimum `loss_correct` on the single
validation family, following the existing Fitzpatrick selection script; test metrics never
select a checkpoint.

The split seed is 20260917. Hashes are shuffled once per fold. Target supports are nested
prefixes across K_T; target query images never enter support. The same episode, support,
query and DDIM noise are used for every arm. Known near-duplicate components are collapsed
before allocation by keeping the lexicographically first hash.

Required arms retain repository meanings:

- correct family + transport: encode matching family source support, apply frozen linear
  transport, no refinement;
- wrong family + relation: encode the other outer-test family's source support, apply the
  same linear transport, no refinement;
- population only: existing `relation_only` strategy with source coordinate zeroed;
- target only: encode K_T target images without source; N/A at K_T=0;
- source reuse: literal source coordinate with no transport or refinement.

### P1 outcomes and gate

The primary comparison is correct-family versus wrong-family at K_T=0. The primary metric
is frozen family semantic consistency (share classified as the requested family; higher is
better). The evaluator is a nine-way TinyCNN using `nine_partition_label`, trained with the
existing instrument recipe after excluding all target-query hashes. It is considered usable
only if its independent held-out top-1 accuracy is at least 40% (chance 11.1%) and each outer
test family has held-out accuracy at least 25%. If this validity rule fails, the semantic gate
is inconclusive regardless of distributional distances.

Let delta_semantic = consistency(correct) - consistency(wrong). A five-percentage-point
absolute gain is the minimum scientifically meaningful effect, fixed before generation:

- clearly positive: point estimate >= 0.05 and the task-level paired 95% CI lower bound > 0;
- source-insensitive: the paired 95% CI upper bound < 0.05;
- negative: the paired 95% CI upper bound < 0;
- otherwise inconclusive.

The six outer-test family means are the independent units. The CI follows the existing
reporting convention: mean +/- 1.96 times the sample standard error over family means.
Secondary lower-is-better metrics are phototype-statistic distance, sliced Wasserstein and
energy MMD. Phenotype target share is reported separately from semantic consistency.
Secondary comparisons are comparator distance minus correct distance, so positive favours
correct transport. No secondary metric can override the semantic primary gate.

## P2: hierarchical wrong-source control

Use the validation-selected frozen disease-level additive checkpoint and its original split,
episodes, K_T grid {0,1,2,5,10,20}, seed 4321 and generation settings. Map diseases to
`nine_partition_label` from the official CSV. For each test disease, choose the alphabetically
next eligible held-out disease in the same family; if none exists, mark that disease's
same-family contrast not estimable. The different-family wrong source is the alphabetically
first eligible held-out disease in a different family, cycled deterministically to avoid one
source being used for every target. Report correct vs same-family wrong, same-family wrong vs
different-family wrong, and each arm vs population-only. No equivalence claim is made without
an equivalence margin; unresolved differences are described as unresolved.

## P3: source-only task separability

Use ImageNet-pretrained ResNet-18 penultimate features with official weights, deterministic
resize/centre-crop preprocessing, per-image L2 normalisation, then a mean vector per disease.
Only clean source-population (Fitzpatrick I--III) images are used. The per-task separation
score is Euclidean distance to the deterministic wrong source assigned in P2. The association
is Spearman rank correlation between this score and the held-out correct-minus-wrong primary
effect, with a task-level bootstrap CI. No encoder or normalisation choice may be changed after
viewing the association. If official weights are unavailable, report P3 blocked rather than
substitute a result-driven representation.

## P4: stronger phenotype separation

Trigger only if P1 is source-insensitive, negative or inconclusive and pre-outcome counts
remain adequate. Source stays I--III; target is fixed once as V--VI. Eligible-family thresholds
remain 64 source and 45 target images. With five eligible families, use five deterministic
folds: test pair (i, i+1 mod 5), validation family (i+2 mod 5), and the remaining two for
training. Average the two outer-test appearances per family before task-level inference.
All method, sampling, metric and gate definitions remain those of P1.

## P5 and P6

The 25% CIFAR task-scarcity run already exists and must be audited, not repeated. Directly
measured timing/memory may be added only with matched warm-up and repetitions. Downstream
demographic augmentation is run only if the P1 primary gate is clearly positive; otherwise
record `NOT RUN -- GATE NOT MET`.

