# P2 hierarchical wrong-source diagnostic

Status: **completed; no robust hierarchical ordering is supported.** Full absolute tables,
paired differences and per-task values are in `../metrics/p2_hierarchical.{md,json}`.

The frozen disease-level checkpoint at step 11,000 was evaluated on the original four test
diseases. Wrong diseases were selected deterministically from held-out validation/test tasks
using the official `nine_partition_label`, before generation. All arms shared target queries,
support draws and sampling noise.

At K_T=0 (positive favours the first named arm):

| Metric | Correct vs same-family wrong | Same-family wrong vs different-family wrong |
|---|---:|---:|
| Semantic consistency | +0.00000 +/- 0.00104 | -0.00065 +/- 0.00074 |
| Phototype-statistic distance | -0.00101 +/- 0.00777 | +0.00053 +/- 0.00532 |
| Sliced Wasserstein | +0.00009 +/- 0.00062 | +0.00004 +/- 0.00038 |
| Energy MMD | +0.00365 +/- 0.01937 | -0.00219 +/- 0.01319 |

No interval in this hierarchy excludes zero. The 16-way semantic evaluator achieved 20.8%
overall held-out accuracy, but only 53.1%, 0%, 6.2% and 0% on the four test diseases in the
zero-shot run. Semantic null differences are therefore not evidence of equivalence.

Conclusion: neither `correct > same-family wrong > different-family wrong` nor a family-only
ordering is resolved. P2 supplies no supported source-semantic interpretation for the frozen
disease model.

