# Audit discrepancies and required corrections

| Item | Earlier report | Audit finding | Revision |
|---|---|---|---|
| Protocol terminology | “pre-registered” / “registered stop rule” | Local creation times show the protocol/config files preceding saved outcomes, but there is no immutable external registration | Replaced by “documented fixed protocol” and disclosed local-timestamp limit |
| P4 phenotype evaluator | Overall 81.2--92.7% described as useful, chance 50% | Real-image class-conditional sanity check finds target recall 0--6.3% and balanced accuracy 50.0--53.1%; aggregate accuracy is dominated by the abundant source class | Mark phenotype target share unreliable for target-fidelity claims |
| P1 phenotype target share | Near-zero values presented as a valid measured fidelity outcome | Same deterministic sanity recipe predicts the source class for every real P1 target image in all three folds | Retain historical values but classify phenotype fidelity as unvalidated |
| Fitzpatrick interpretation | “learns a shared phenotype shift/offset” | Energy MMD improves over population-only, but phenotype and semantic identity are not validated | Restrict claim to a distributional shift on energy MMD |
| Correct vs wrong | “statistically indistinguishable” | A CI covering zero is not an equivalence test | Use “no resolved difference” / “inconclusive” |
| P4 causal reading | Larger phenotype gap does not recover specificity | P4 changes both target definition and eligible-family composition (6 to 5) | Describe P4 as a second, differently composed cohort; do not isolate the cause |
| Linear transport | “not worse” and approximately linear | No pre-fixed non-inferiority margin; linear has lower SW and MMD on tested FiLM budgets | State the observed matched result only; avoid global geometry claim |
| P5 completeness | Compressed prose omitted raw values/CIs for several claims | Existing artifacts contain the missing paired evidence | Add compact raw/effect tables and a traceable evidence appendix |
| Efficiency | Parameter count presented beside general efficiency language | Matched FLOPs, latency and peak memory were never measured | Parameter-efficiency only; mark other measures NOT MEASURED |

No arithmetic discrepancy was found in the published P1--P4 aggregate cells.  The substantive
correction concerns interpretation and evaluator validity, not selective recomputation of model
outputs.

