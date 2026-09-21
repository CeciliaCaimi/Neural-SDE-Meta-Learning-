# P1 pathology-family semantic resolution

Status: **completed; locked verdict inconclusive because the semantic evaluator failed its
validity rule.** The complete absolute and paired tables for every method and every budget
are in `../metrics/p1_family_aggregate.md`; machine-readable values are in the adjacent JSON.

Three task-level folds covered six eligible families exactly once. Validation-only selection
chose steps 16,000, 10,000 and 10,000. All evaluations used linear transport, no target-time
refinement, K_T={0,1,2,5,10,20}, 96 samples, DDIM-50 and eight paired repeats.

| Fold | Test families | Overall evaluator accuracy | Worst test-family accuracy | Valid |
|---:|---|---:|---:|:---:|
| 0 | benign dermal; benign epidermal | 0.676 | 0.000 | no |
| 1 | genodermatoses; inflammatory | 0.683 | 0.000 | no |
| 2 | malignant epidermal; malignant melanoma | 0.687 | 0.153 | no |

At K_T=0, correct-family minus wrong-family semantic consistency was 0.0000 +/- 0.0000,
but this zero is uninformative because the classifier could not recognise several real
held-out families. The valid secondary paired effects (positive favours correct transport)
were: phototype-statistic distance -0.00029 +/- 0.00160, sliced Wasserstein
+0.00007 +/- 0.00091, and energy MMD +0.00317 +/- 0.01925. None excludes zero. Since the
headline correct/wrong arms use no target-time refinement, their coordinate is independent
of K_T and these effects repeat across the nonzero target budgets by design.

Conclusion: the experiment does not establish that family-level source identity helps.
It also cannot prove equivalence from the failed semantic instrument. The locked outcome is
inconclusive, with independent distributional diagnostics showing no resolved source-specific
advantage.

