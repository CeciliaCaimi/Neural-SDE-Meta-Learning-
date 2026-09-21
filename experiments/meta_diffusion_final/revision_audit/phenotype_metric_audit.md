# Exploratory audit of phenotype target share

## Exact metric

`phenotype_target_share` is the fraction of 32x32 generated images for which a binary TinyCNN
hard-argmax prediction is class 1.  Class 0 is source phototypes I--III; class 1 is target
phototypes IV--VI (P1) or V--VI (P4).  Generated tensors and real images are both supplied in
RGB NCHW `[-1,1]`.  There is no confidence threshold, abstention or calibration step.

The evaluator is trained on train/validation-condition images after withholding all test
conditions and target-query hashes.  However, the recipe samples the naturally imbalanced
data, reports only aggregate accuracy, and did not explicitly seed PyTorch before TinyCNN
initialization.  The saved evaluator weights are not archived.

## Exploratory real-image sanity check

`scripts/audit_fitz_phenotype_instrument.py` reruns the unchanged training recipe with an
explicit torch seed (4321) and evaluates every real source/target image of the frozen outer
test families.  This is post-outcome and diagnostic; it does not replace confirmatory cells.

| Protocol/fold | Aggregate test accuracy | Real-source recall | Real-target recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| P1 / 0 | 0.780 | 1.000 | 0.000 | 0.500 |
| P1 / 1 | 0.727 | 1.000 | 0.000 | 0.500 |
| P1 / 2 | 0.811 | 1.000 | 0.000 | 0.500 |
| P4 / 0 | 0.904 | 1.000 | 0.000 | 0.500 |
| P4 / 1 | 0.813 | 1.000 | 0.005 | 0.503 |
| P4 / 2 | 0.857 | 1.000 | 0.000 | 0.500 |
| P4 / 3 | 0.880 | 0.998 | 0.063 | 0.531 |
| P4 / 4 | 0.927 | 1.000 | 0.009 | 0.505 |

The apparently high aggregate accuracies are compatible with near-constant source-class
prediction because source images dominate.  Consequently, the near-zero generated target
shares cannot establish phenotype failure or success: the instrument itself has essentially
no target-class sensitivity in this sanity check.  There is no validated phenotype-fidelity
success result for P1 or P4.  Fixed phototype statistics, SW and MMD remain distributional
diagnostics and must not be described as proof of phenotype identity.

