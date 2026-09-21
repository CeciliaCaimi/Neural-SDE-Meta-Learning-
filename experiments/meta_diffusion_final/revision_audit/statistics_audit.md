# Statistical and implementation audit

## Independent units and intervals

- P1: eight repeats are averaged inside each of six outer-test families; inference uses the
  six family means.
- P2/P3: inference uses four disease tasks.  This is too small for a broad generalization.
- P4: each family appears in two cyclic outer folds; the two appearances are averaged first,
  then inference uses five family means.
- Reported intervals are mean +/- `1.96 * sample SD / sqrt(n)`.  They are normal-approximation
  95% CI half-widths, not standard deviations and not exact small-sample t intervals.
- A star is attached only when the paired interval excludes zero.  This is not an equivalence
  test and does not address multiplicity across the large diagnostic grid.

## Pairing and target-only path

`scripts/c4_scarcity.py` resets support/query draws and generation noise per task/repeat;
comparisons within a cell are matched.  Nested target support prefixes are used across K.
`adaptation/coordinate.py` confirms that `target_only` computes
`encoder(batch.tgt_support)` and then refines that coordinate on the same target support.
At K=0 target-only is omitted as N/A.  No-refinement transport, wrong-source, relation-only,
and literal source-reuse arms do not consume target images and are therefore budget-invariant
by construction.  The weak target-only K trend is an observed outcome, not evidence that K
was ignored.

## Metric directions

Phenotype target share and semantic consistency are higher-is-better.  Fixed phototype
signature distance, sliced Wasserstein and energy MMD are lower-is-better.  For distances the
report uses comparator minus correct transport; for consistency it uses correct transport
minus comparator, so positive always favors correct transport.

## Scope limits

Pooled KID was skipped for Fitzpatrick because the semantic instrument failed and the relevant
sample/task support was inadequate for a meaningful headline estimate.  P3's Spearman result
is descriptive: n=4, three effects are tied at zero, and only 6,793/10,000 bootstrap draws have
defined rank variance.

