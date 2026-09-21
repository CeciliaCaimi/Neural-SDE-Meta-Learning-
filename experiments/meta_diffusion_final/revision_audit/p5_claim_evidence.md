# P5 CIFAR claim evidence

All intervals below are paired 95% CI half-widths from the saved artifact's established
episode-level method.  SW and MMD are lower-is-better; class consistency is higher-is-better.
An asterisk means the paired interval excludes zero; it is not an equivalence claim.

## Zero-shot source specificity (A6; n=60 held-out episodes)

| K | Metric | Correct raw | Wrong raw | Correct advantage |
|---:|---|---:|---:|---:|
| 0 | SW | 0.210 +/- 0.020 | 0.248 +/- 0.021 | +0.0386 +/- 0.0141* |
| 0 | Fine-class consistency | 0.111 +/- 0.023 | 0.052 +/- 0.010 | +0.0597 +/- 0.0204* |

Artifact: `handoff/results/A6_zero_shot.{json,txt}`.  Generation: 96 images/cell,
DDIM-50, source budget 64.

## FiLM source controls (A7; matched held-out episodes)

Each effect is comparator SW minus correct-transport SW, so positive favors correct transport.

| Comparator | K=1 raw / effect | K=5 raw / effect | K=20 raw / effect |
|---|---:|---:|---:|
| Correct transport | 0.207 +/- 0.022 / -- | 0.205 +/- 0.019 / -- | 0.207 +/- 0.021 / -- |
| Mean source | 0.246 +/- 0.024 / +0.0397 +/- 0.0084* | 0.248 +/- 0.020 / +0.0430 +/- 0.0091* | 0.250 +/- 0.023 / +0.0430 +/- 0.0096* |
| Shuffled source | 0.257 +/- 0.025 / +0.0501 +/- 0.0154* | 0.259 +/- 0.023 / +0.0533 +/- 0.0169* | 0.261 +/- 0.024 / +0.0537 +/- 0.0167* |
| Relation only | 0.241 +/- 0.023 / +0.0342 +/- 0.0077* | 0.242 +/- 0.020 / +0.0361 +/- 0.0082* | 0.245 +/- 0.023 / +0.0380 +/- 0.0090* |

Fine-class correct-transport raw values are 0.125 +/- 0.026, 0.125 +/- 0.028 and
0.125 +/- 0.027 at K=1,5,20.  Its paired advantages over mean, shuffled and relation-only
are respectively 0.0708 +/- 0.0226*, 0.0724 +/- 0.0221*, 0.0696 +/- 0.0211* at K=1;
0.0717 +/- 0.0256*, 0.0714 +/- 0.0254*, 0.0773 +/- 0.0245* at K=5; and
0.0708 +/- 0.0235*, 0.0729 +/- 0.0244*, 0.0719 +/- 0.0217* at K=20.

## Linear versus nonlinear FiLM transport (A8)

Effect is nonlinear minus linear for lower-is-better metrics; positive favors linear.

| Metric | K | Nonlinear raw | Linear raw | Linear advantage |
|---|---:|---:|---:|---:|
| SW | 1 | 0.2066 +/- 0.0224 | 0.1973 +/- 0.0206 | +0.0093 +/- 0.0037* |
| SW | 5 | 0.2054 +/- 0.0185 | 0.1942 +/- 0.0164 | +0.0112 +/- 0.0040* |
| SW | 20 | 0.2074 +/- 0.0214 | 0.1987 +/- 0.0193 | +0.0087 +/- 0.0042* |
| Energy MMD | 1 | 3.0686 +/- 0.7686 | 2.8207 +/- 0.6413 | +0.2479 +/- 0.1658* |
| Energy MMD | 5 | 2.9091 +/- 0.5972 | 2.6281 +/- 0.4690 | +0.2810 +/- 0.1588* |
| Energy MMD | 20 | 3.1186 +/- 0.8714 | 2.9228 +/- 0.7610 | +0.1958 +/- 0.1640* |

This supports a narrow statement about the tested FiLM checkpoints and budgets, not a proof
that task-coordinate geometry is globally linear.

## Additive basis versus FiLM (A2; n=180 validation episodes; one training seed/arm)

Transport SW, basis versus FiLM, is 0.2064/0.2066 at K=1, 0.2074/0.2054 at K=5 and
0.2128/0.2074 at K=20.  Paired FiLM-minus-basis effects are +0.0002 +/- 0.0070,
-0.0020 +/- 0.0076 and -0.0055 +/- 0.0074: none is resolved as different.  Conditioning
parameter counts are 55,344 (basis) and 118,784 (FiLM).  Latency, FLOPs and peak memory are
NOT MEASURED.

## Full fine-tuning reference (A9; n=180 paired episodes)

Effect is comparator SW minus transport SW; positive favors transport.

| K | Target-only raw | Transport raw | Full-FT raw | Target-only effect | Full-FT effect |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2027 +/- 0.0139 | 0.1839 +/- 0.0122 | 0.2675 +/- 0.0193 | +0.0189 +/- 0.0092* | +0.0836 +/- 0.0180* |
| 5 | 0.1950 +/- 0.0148 | 0.1896 +/- 0.0120 | 0.2124 +/- 0.0183 | +0.0054 +/- 0.0069 | +0.0228 +/- 0.0202* |
| 20 | 0.1840 +/- 0.0111 | 0.1896 +/- 0.0122 | 0.1465 +/- 0.0091 | -0.0056 +/- 0.0067 | -0.0431 +/- 0.0149* |

## 25% meta-training tasks (C7; n=180 validation episodes; one training seed/arm)

Effect is FiLM minus basis for lower-is-better metrics; positive favors basis.

| Metric | K | Basis raw | FiLM raw | Basis advantage |
|---|---:|---:|---:|---:|
| SW | 1 | 0.2005 +/- 0.0191 | 0.2187 +/- 0.0245 | +0.0181 +/- 0.0095* |
| SW | 5 | 0.1957 +/- 0.0163 | 0.2160 +/- 0.0214 | +0.0203 +/- 0.0093* |
| SW | 20 | 0.2017 +/- 0.0168 | 0.2215 +/- 0.0219 | +0.0198 +/- 0.0091* |
| Energy MMD | 1 | 2.9180 +/- 0.5704 | 3.3848 +/- 0.8191 | +0.4668 +/- 0.3434* |
| Energy MMD | 5 | 2.8354 +/- 0.4936 | 3.3077 +/- 0.7547 | +0.4723 +/- 0.3790* |
| Energy MMD | 20 | 3.1129 +/- 0.5783 | 3.6103 +/- 0.7813 | +0.4975 +/- 0.3429* |

Fine-class consistency does not resolve a basis advantage at any budget.  These are conditional
one-seed observations; they do not establish universal basis superiority.

## 50% meta-training tasks and direction reversal

At 50%, basis/FiLM SW is 0.2097/0.1912, 0.2060/0.1851 and 0.2141/0.1920 at
K=1,5,20.  FiLM-minus-basis effects are -0.0185 +/- 0.0108*, -0.0208 +/- 0.0114* and
-0.0221 +/- 0.0122*, favoring FiLM.  The reversal relative to the one-seed 25% run is direct
evidence that task-scarcity architecture rankings are not stable enough for a general claim.

## Fitzpatrick optional representation-strength checks

The 50% task experiment uses the older four-disease split.  FiLM has lower fixed-statistic,
SW and energy-MMD distances than basis at each reported budget; for SW the FiLM advantages
are 0.0214 +/- 0.0048* at K=1 and 0.0199 +/- 0.0057* at K=20.  The coordinate-compression
run finds basis k=4 lower than k=16 on SW by 0.0030 +/- 0.0027*, 0.0053 +/- 0.0031* and
0.0038 +/- 0.0017* at K=1,5,20.  Both experiments use one training seed per arm/dimension and
four disease tasks; they are distributional diagnostics, not validated phenotype or semantic
fidelity results.  Artifacts: `C7_fitz_train50_representation.{txt,json}` and
`C7_fitz_coordinate_compression.txt`.
