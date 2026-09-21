# Evidence manifest for the 2026-09-21 report revision

Revision baseline: branch `jinjian`, report commit
`2cd24b1aa79f8425c9d361c65d838751862a26e9`.  The experiment protocol records repository
HEAD `6de5187426b4ebfb4545846a49ca4d30f02987cb` at lock time.  Status terms are reporting-audit
statuses, not new experimental outcomes.

| Claim ID | Report location | Primary evidence | Aggregation / unit | Status and edit |
|---|---|---|---|---|
| P0-DATA | P0 | `prerequisite_audit.md`; P1/P4 manifests | official hashes and deduplicated image components | VERIFIED; add official CSV/manifest hashes and exclusions |
| P0-PROTOCOL | P0 | `protocol_lock.md`; P1/P4 configs; local timestamps | protocol file precedes first saved output locally | CORRECTED; use “documented fixed protocol,” not “pre-registered” |
| P1-PRIMARY | P1 tables | `metrics/p1_family_aggregate.json` SHA-256 `4B55...9B59` | six outer-test family means; repeats averaged within family; mean +/- 1.96 SE | VERIFIED; semantic conclusion remains INCONCLUSIVE |
| P1-SPECIFICITY | P1 tables | same artifact, correct-vs-wrong rows | six paired family effects | VERIFIED; no resolved correct-vs-wrong difference |
| P1-PHENOTYPE | P1 tables | saved generated shares plus evaluator audit | saved hard-label shares; exploratory real-image sanity check | CORRECTED; metric is not validated for phenotype fidelity |
| P2-HIERARCHY | P2 tables | `metrics/p2_hierarchical.json` SHA-256 `8662...0FC` | four paired disease effects | VERIFIED; hierarchy unresolved, semantic evaluator weak |
| P3-SEPARABILITY | P3 | `metrics/p3_task_separability.json` SHA-256 `28D5...50D` | four tasks; 6,793/10,000 defined bootstrap correlations | VERIFIED; descriptive/fragile only |
| P4-PRIMARY | P4 tables | `metrics/p4_v_vi_aggregate.json` SHA-256 `BBC5...73A` | five family means after averaging two cyclic appearances | VERIFIED; semantic conclusion remains INCONCLUSIVE |
| P4-COMPOSITION | P4 discussion | P1/P4 manifests | P1 has six families, P4 five | CORRECTED; between-study change cannot be assigned solely to phenotype gap |
| P4-PHENOTYPE | P4 tables | saved generated shares; `phenotype_metric_audit.md` | exploratory class-conditional real-image check | CORRECTED; prior 81--93% overall accuracy was imbalance-sensitive |
| P5-K0 | P5 | `handoff/results/A6_zero_shot.{json,txt}` SHA-256 JSON `1AE3...D747` | 60 paired held-out episodes | VERIFIED; SW and fine-class effects shown with raw values and CI |
| P5-FILM | P5 | `A7_film_source_dependence.{json,txt}` SHA-256 JSON `958E...CD6B` | matched held-out episodes at K=1,5,20 | VERIFIED; complete SW effects added |
| P5-LINEAR | P5 | `A8_film_linear_per_episode.json`, `A8_film_linear_vs_nonlinear.txt` | matched episodes at K=1,5,20 | CORRECTED; linear has lower SW/MMD on these budgets; no global linearity proof |
| P5-BASIS-FILM | P5 | `A2_basis_vs_film.txt` | 180 paired validation episodes, one training seed/arm | VERIFIED; SW differences unresolved; parameter counts are measured |
| P5-FT | P5 | `A9_cifar_full_ft.{json,txt}` SHA-256 JSON `DB85...AEA` | 180 paired episodes | VERIFIED; full raw and paired SW values added |
| P5-25 | P5 | `C7_cifar_train25_*_per_episode.json`; comparison text | 180 paired validation episodes, one training seed/arm | VERIFIED WITH LIMITATION; conditional one-seed observation |
| P5-EFF | P5 | logs and report audit | no matched benchmark artifact | NOT MEASURED; no latency/FLOPs/memory inference |
| P6-GATE | P6 | `protocol_lock.md`; P1 validity and primary gate | fixed semantic trigger | VERIFIED; NOT RUN -- GATE NOT MET |

Full absolute cells, per-family effects, checkpoint records, seeds and commands remain in the
referenced JSON/TXT files.  No result value was silently replaced during this revision.

