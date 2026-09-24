# GMM boundary study — results (Jing / training)

Controlled 2-D GMM phase-diagram study of when source-informed transport helps.
Grid: 7 alpha (source identifiability) x 6 eta (relation heterogeneity), 5 seeds,
K_T=0, 8000 steps. SageMaker job jp-gmmlid-20260923-114734 (210/210 cells, 10.3 GPU-h).

Primary metric: G_source = SW(wrong-source) - SW(correct-source), per test task,
averaged over tasks and seeds. Positive => using the correct source helps.

Diagnostics (independent of Meta-Diff):
- bayes_id    : Bayes task-ID accuracy from TRUE GMM likelihoods (intrinsic identifiability).
- learned_id  : nearest-centroid task-ID accuracy in the TRAINED set-encoder's z_S space
                (usable identifiability). Same MS-source protocol + test tasks as bayes_id;
                chance = 1/64 = 0.0156.
- collapse_gap = bayes_id - learned_id (recoverable identity the encoder discards).
- oracle_r2   : held-out R^2 of an oracle affine source->target mean map (relation predictability).

## Files
- gmm_boundary_learnedid_report.txt   full text report (heatmaps, regressions, section-12 gates)
- tableC_gmm_boundary_learnedid.json   per-cell table (bayes_id, learned_id, gap, oracle_r2, G_source + 95% CI)
- fig_heatmap_Gsource.png              G_source phase diagram
- fig_heatmap_learned_id.png           learned-space task-ID
- fig_heatmap_collapse_gap.png         Bayes-ID minus learned-ID
- fig_collapse_curve.png               intrinsic vs usable identity across alpha (the gap)
- fig_mechanism_bayes_vs_learned.png   HEADLINE: G_source vs Bayes-ID (rho=0.69) vs learned-ID (rho=0.98)
- rows/gmm_rows_*.jsonl                raw per-cell / per-task records

## Headline findings
1. Mechanism holds: G_source rises with identifiability, falls with relation heterogeneity;
   at alpha=0 every CI contains 0, for alpha>=0.1 essentially every CI excludes 0.
2. Collapse gap peaks at intermediate alpha=0.25 (bayes 0.95, learned 0.33): the task is
   identifiable in principle but the learned coordinate captures only ~1/3 of it. eta does
   not move learned_id (0.52-0.54), so it is a clean source-encoder property.
3. The benefit tracks USABLE identity, not intrinsic: rho(G,learned)=0.98 vs rho(G,Bayes)=0.69;
   regression R^2 0.84 vs 0.59; in the combined model learned-ID is +0.73 while Bayes-ID is
   -0.27. Restricted to fully-identifiable cells (Bayes>0.99), rho(G,learned)=0.97.

Not yet run: K_T in {1,5}, M_S in {8,32,128}, kappa transport-nonlinearity sweeps.
