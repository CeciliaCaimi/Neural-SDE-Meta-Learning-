# Final synthesis for the executable experiment plan

This document separates newly executed evidence from audited prior evidence. Exact commands,
seeds and checkpoint selections are recorded in `../runs/execution_log.md`; the frozen
decisions and direction of every comparison are in `../protocol_lock.md`.

## Evidence and decision table

| Priority | Plan question | Status and evidence-backed answer |
|---|---|---|
| P0 | Are data, splits, checkpoints and inference units suitable? | **Passed.** The verified Fitzpatrick17k archive, hashes, exclusions, task-level folds, source/target definitions, frozen code and CIFAR artifacts were audited before evaluation. |
| P1 | Does source specificity reappear for pathology families (I--III to IV--VI)? | **Inconclusive primary / no resolved secondary advantage.** The semantic evaluator failed its locked per-family validity rule. Correct-vs-wrong effects were also unresolved on statistic distance, SW, energy MMD and phenotype share. |
| P2 | Is there a correct > same-family wrong > different-family wrong hierarchy? | **No robust ordering.** All principal paired intervals crossed zero; the semantic instrument was weak, so null differences are not equivalence. |
| P3 | Does source-only semantic separability predict the correct-source advantage? | **Unsupported and unresolved.** Spearman rho=-0.775 with bootstrap interval [-1.000,-0.775] among defined resamples, but n=4, three effects were exactly zero and the effect itself came from an invalid semantic evaluator. This cannot support a mechanistic claim. |
| P4 | Does a stronger I--III to V--VI phenotype gap recover source specificity? | **No.** Correct transport beat population-only on energy MMD (+0.988 +/- 0.708), but not wrong-task transport (-0.001 +/- 0.039); the semantic primary remained invalid. Stop demographic tuning. |
| P5 | Is the frozen CIFAR package complete? | **Complete from audited saved evidence.** CIFAR supports source-dependent K_T=0 transfer, FiLM generality, a linear relation, the low-data/full-fine-tuning crossover, and a one-seed 25%-task basis advantage. Basis-vs-FiLM generation quality remains unresolved; basis is a parameter-efficiency result. |
| P6 | Should downstream demographic augmentation run? | **NOT RUN -- GATE NOT MET.** P1 was not clearly positive under its locked rule. |

## Main quantitative findings

On P1 (six families), correct-minus-wrong semantic consistency at K_T=0 was
0.0000 +/- 0.0000 but invalid as primary evidence. Valid secondary paired effects were
statistic distance -0.00029 +/- 0.00160, SW +0.00007 +/- 0.00091, energy MMD
+0.00317 +/- 0.01925 and phenotype share +0.00022 +/- 0.00043; none excluded zero.

On P2 (four held-out diseases), no semantic or distributional hierarchy was resolved.
At K_T=0, correct-vs-same-family and same-vs-different-family effects were respectively:
semantic 0.00000 +/- 0.00104 and -0.00065 +/- 0.00074; statistic distance
-0.00101 +/- 0.00777 and +0.00053 +/- 0.00532; SW +0.00009 +/- 0.00062 and
+0.00004 +/- 0.00038; energy MMD +0.00365 +/- 0.01937 and -0.00219 +/- 0.01319.

On P4 (five families), the correct-vs-wrong effects at K_T=0 were phenotype share
-0.0005 +/- 0.0007, statistic distance -0.0007 +/- 0.0027, SW -0.0000 +/- 0.0013,
and energy MMD -0.0011 +/- 0.0391. None was significant. Correct transport did improve
energy MMD over population-only by +0.9878 +/- 0.7085 (positive favours correct), but this
is evidence for a shared population shift rather than correct-task source specificity.

Audited CIFAR evidence remains the positive core: at K_T=0, correct transport improved SW
over wrong source by 0.0386 +/- 0.0141 and fine-class consistency by 0.0597 +/- 0.0204;
both intervals exclude zero. The same source-dependent phenomenon survives FiLM, and linear
transport is adequate. At 25% of meta-training tasks, the frozen one-seed basis run beat FiLM
on SW by 0.0181--0.0203 and energy MMD by 0.4668--0.4975 across K_T=1,5,20.

## Claim boundary

The verified manuscript claim is narrow but coherent:

> On CIFAR, correct source identity supports an approximately linear reusable task-space
> relation, is most useful under extreme target scarcity, and generalises across additive
> basis and FiLM conditioning. The additive basis is compact, not demonstrably superior in
> generation quality.

Fitzpatrick17k does **not** currently extend that claim to disease- or pathology-family-
specific demographic transport. Across the original family protocol, hierarchical controls,
source-only separability diagnostic and stronger V--VI shift, the correct and wrong source
arms are unresolved and often numerically identical. The training diagnostics independently
show that a shared mean coordinate captures nearly all coordinate gain. The defensible
interpretation is a real-data limitation: the frozen model learns a largely shared phenotype
offset, while disease-specific task identity is not demonstrated. The invalid semantic
instrument prevents turning the null into an equivalence claim.

## Reproducibility and artifacts

- Protocol and directionality: `../protocol_lock.md`
- Dataset/environment audit: `../prerequisite_audit.md`
- P1 full tables: `../metrics/p1_family_aggregate.md`
- P2 full tables: `../metrics/p2_hierarchical.md`
- P3 scores and association: `../metrics/p3_task_separability.json`
- P4 full tables: `../metrics/p4_v_vi_aggregate.md`
- P5 audited files and conclusions: `p5_cifar_completion.md`
- Exact execution record: `../runs/execution_log.md`

No P6 results exist because its strict gate failed. Debug runs are named explicitly and were
excluded from all reported estimates.
