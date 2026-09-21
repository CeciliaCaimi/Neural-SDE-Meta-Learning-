# P4 stronger phenotype separation: Fitzpatrick I--III to V--VI

Status: **completed; the source-specific result remains weak and the locked primary
semantic verdict is inconclusive because the evaluator failed its validity rule.**

The protocol was fixed before P1 was inspected. Five eligible pathology families were
evaluated in five cyclic task-level folds, with each family appearing in two outer test
folds and averaged once before inference. Validation-only selection chose steps 16,000,
4,000, 4,000, 4,000 and 10,000. The frozen architecture was the additive-basis
small U-Net (0.72M backbone parameters), set encoder (0.33M), linear transport (0.3K),
coordinate dimension 16, source support 32, and no target-time refinement. Evaluation used
K_T={0,1,2,5,10,20}, 96 samples per cell, DDIM-50, eight paired repeats and seed 4321.

## Locked evaluator check

| Fold | Overall held-out accuracy | Worst test-family accuracy | Valid |
|---:|---:|---:|:---:|
| 0 | 0.701 | 0.000 | no |
| 1 | 0.699 | 0.000 | no |
| 2 | 0.704 | 0.000 | no |
| 3 | 0.705 | 0.127 | no |
| 4 | 0.701 | 0.025 | no |

The phenotype instrument was useful on the outer test families (81.2--92.7% across folds;
chance 50%), but the five-way family instrument was not. Accordingly, family semantic
consistency is reported but cannot support the headline claim. Pooled KID was also skipped
under the locked feature-validity rule.

## Zero-shot absolute performance

Values are family means +/- 95% confidence-interval half-widths (n=5 independent families).
For phenotype share and semantic consistency, higher is better. For statistic distance,
sliced Wasserstein (SW), and energy MMD, lower is better.

| Method | Phenotype share | Family consistency | Statistic distance | SW | Energy MMD |
|---|---:|---:|---:|---:|---:|
| Source reuse | 0.0311 +/- 0.0359 | 0.2033 +/- 0.3807 | 1.8099 +/- 0.2613 | 0.2649 +/- 0.0446 | 4.5031 +/- 0.7991 |
| Population only | 0.0000 +/- 0.0000 | 0.1997 +/- 0.3909 | 2.2577 +/- 0.4548 | 0.2527 +/- 0.0275 | 5.4122 +/- 1.1998 |
| Wrong-task source + relation | 0.0320 +/- 0.0369 | 0.2033 +/- 0.3807 | 1.7943 +/- 0.2765 | 0.2637 +/- 0.0454 | 4.4233 +/- 0.8218 |
| Correct source + transport | 0.0315 +/- 0.0363 | 0.2031 +/- 0.3808 | 1.7950 +/- 0.2769 | 0.2638 +/- 0.0461 | 4.4244 +/- 0.8362 |

## Zero-shot paired comparisons

All entries are oriented so **positive values favour correct source + transport**. A star
means that the paired 95% interval over family-level mean differences excludes zero.

| Comparator | Phenotype share | Family consistency | Statistic distance | SW | Energy MMD |
|---|---:|---:|---:|---:|---:|
| Source reuse | +0.0004 +/- 0.0005 | -0.0001 +/- 0.0010 | +0.0149 +/- 0.0162 | +0.0012 +/- 0.0027 | +0.0787 +/- 0.0827 |
| Population only | +0.0315 +/- 0.0363 | +0.0034 +/- 0.0161 | +0.4627 +/- 0.4693 | -0.0111 +/- 0.0238 | +0.9878 +/- 0.7085* |
| Wrong-task source + relation | -0.0005 +/- 0.0007 | -0.0001 +/- 0.0003 | -0.0007 +/- 0.0027 | -0.0000 +/- 0.0013 | -0.0011 +/- 0.0391 |

The only resolved zero-shot aggregate contrast is energy MMD versus population-only. The
correct-versus-wrong-task comparison is unresolved for every valid secondary metric and is
numerically near zero. The semantic primary effect is -0.0001 +/- 0.0003, but is
uninformative because the semantic evaluator failed the locked rule.

Because the headline method has no target-time refinement, correct source, wrong source,
source reuse and population-only coordinates do not change with K_T. The nonzero budgets
therefore measure target-only data efficiency, not additional adaptation of those arms.
Target-only and correct transport have overlapping paired intervals on SW and energy MMD at
all K_T values; the closest point estimates do not constitute a formal equivalence result.
The complete absolute and paired tables for every budget are in
`../metrics/p4_v_vi_aggregate.md`, with machine-readable values in the adjacent JSON.

## Conclusion

Increasing the phenotype gap from IV--VI to V--VI does not recover a disease-family-specific
source advantage. The run supports a reusable population shift relative to the relation-only
control on energy MMD, but the same shift is obtained with a wrong source task. Under the
pre-registered stop rule, demographic model tuning stops here; no architecture rescue or
downstream augmentation experiment is justified.
