# P5 CIFAR completion audit

Status: **complete from saved, inspected evidence; no rerun required.** The outputs named
below contain the absolute episode-level metrics and paired 95% intervals. All use matched
episodes and sampling noise.

| Plan question | Saved evidence | Result supported |
|---|---|---|
| Explicit K_T=0 transfer | `handoff/results/A6_zero_shot.{txt,json}` | Correct transport has lower SW than wrong source by 0.0386 +/- 0.0141 and higher fine-class consistency by 0.0597 +/- 0.0204; both intervals exclude zero. |
| Conditioning generality | `handoff/results/A7_film_source_dependence.{txt,json}` | Under FiLM, correct transport beats shuffled, mean-source and relation-only controls on SW and semantic consistency at K_T=1,5,20. |
| Linear relation | `handoff/results/A8_film_linear_vs_nonlinear.txt` | Linear transport is not worse; it is significantly better on SW at every reported budget (gains 0.0093, 0.0112 and 0.0087). |
| Basis versus FiLM | `handoff/results/A2_basis_vs_film.txt` | Their transported SW is unresolved as different at K_T=1,5,20. Basis uses 55,344 coordinate-conditioning parameters versus 118,784 for FiLM (46.6%). This is an efficiency, not quality-superiority, claim. |
| Full fine-tuning crossover | `handoff/results/A9_cifar_full_ft.{txt,json}` | Transport is significantly better at K_T=1 and K_T=5; full fine-tuning is significantly better at K_T=20 on SW. |
| 25% meta-training tasks | `handoff/results/C7_cifar_train25_taskscarcity_compare.txt` and paired JSON inputs | At 25%, basis beats FiLM on SW by 0.0181--0.0203 and on energy MMD by 0.4668--0.4975 at K_T=1,5,20; all paired intervals exclude zero. FiLM degrades versus its 100% result, while basis does not. One training seed per arm limits the claim to this frozen run. |

The completed package supports source-dependent zero-shot transfer, architectural generality,
an approximately linear coordinate relation, and a low-data crossover against full
fine-tuning. It does **not** support an additive-basis generation-quality advantage over
FiLM. Matched latency, FLOPs and peak-memory measurements were not present in the saved logs
and are not inferred from parameter counts.
