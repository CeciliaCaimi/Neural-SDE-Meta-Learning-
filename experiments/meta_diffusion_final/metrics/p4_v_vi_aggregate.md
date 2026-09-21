# P4 stronger-phenotype pathology-family result

**Locked primary verdict: inconclusive: semantic evaluator failed the locked validity rule.**

The primary effect is correct-family minus wrong-family semantic consistency at K_T=0; positive favours correct-family transport.

Primary effect: -0.0001 +/- 0.0003 (95% CI half-width; n=5 outer-test families).

## Semantic-instrument validity

| Fold | Overall held-out accuracy | Minimum test-family accuracy | Pass |
|---:|---:|---:|:---:|
| 0 | 0.701 | 0.000 | no |
| 1 | 0.699 | 0.000 | no |
| 2 | 0.704 | 0.000 | no |
| 3 | 0.705 | 0.127 | no |
| 4 | 0.701 | 0.025 | no |

## K_T=0

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.0311 +/- 0.0359 | +0.0004 +/- 0.0005 |
| population only | 0.0000 +/- 0.0000 | +0.0315 +/- 0.0363 |
| wrong-task source + relation | 0.0320 +/- 0.0369 | -0.0005 +/- 0.0007 |
| correct source + transport | 0.0315 +/- 0.0363 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.2033 +/- 0.3807 | -0.0001 +/- 0.0010 |
| population only | 0.1997 +/- 0.3909 | +0.0034 +/- 0.0161 |
| wrong-task source + relation | 0.2033 +/- 0.3807 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.2031 +/- 0.3808 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=1

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0046 +/- 0.0053 | -0.0007 +/- 0.0010 |
| source reuse | 0.0040 +/- 0.0046 | -0.0001 +/- 0.0010 |
| population only | 0.0000 +/- 0.0000 | +0.0039 +/- 0.0047 |
| wrong-task source + relation | 0.0040 +/- 0.0047 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.0039 +/- 0.0047 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2236 +/- 0.3794 | -0.0033 +/- 0.0053 |
| source reuse | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0013 |
| population only | 0.2007 +/- 0.3917 | +0.0197 +/- 0.0244 |
| wrong-task source + relation | 0.2220 +/- 0.3804 | -0.0017 +/- 0.0030 |
| correct source + transport | 0.2203 +/- 0.3811 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.8082 +/- 0.2694 | +0.0132 +/- 0.0109* |
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2597 +/- 0.0413 | -0.0041 +/- 0.0052 |
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 4.3571 +/- 0.7399 | -0.0673 +/- 0.1207 |
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=2

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0044 +/- 0.0051 | -0.0005 +/- 0.0014 |
| source reuse | 0.0040 +/- 0.0046 | -0.0001 +/- 0.0010 |
| population only | 0.0000 +/- 0.0000 | +0.0039 +/- 0.0047 |
| wrong-task source + relation | 0.0040 +/- 0.0047 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.0039 +/- 0.0047 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2225 +/- 0.3801 | -0.0022 +/- 0.0027 |
| source reuse | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0013 |
| population only | 0.2007 +/- 0.3917 | +0.0197 +/- 0.0244 |
| wrong-task source + relation | 0.2220 +/- 0.3804 | -0.0017 +/- 0.0030 |
| correct source + transport | 0.2203 +/- 0.3811 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.8050 +/- 0.2706 | +0.0100 +/- 0.0121 |
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2622 +/- 0.0440 | -0.0016 +/- 0.0022 |
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 4.3970 +/- 0.7715 | -0.0274 +/- 0.0679 |
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=5

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0042 +/- 0.0048 | -0.0003 +/- 0.0013 |
| source reuse | 0.0040 +/- 0.0046 | -0.0001 +/- 0.0010 |
| population only | 0.0000 +/- 0.0000 | +0.0039 +/- 0.0047 |
| wrong-task source + relation | 0.0040 +/- 0.0047 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.0039 +/- 0.0047 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0007 |
| source reuse | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0013 |
| population only | 0.2007 +/- 0.3917 | +0.0197 +/- 0.0244 |
| wrong-task source + relation | 0.2220 +/- 0.3804 | -0.0017 +/- 0.0030 |
| correct source + transport | 0.2203 +/- 0.3811 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.8044 +/- 0.2691 | +0.0094 +/- 0.0086* |
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2624 +/- 0.0443 | -0.0013 +/- 0.0020 |
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 4.4157 +/- 0.7973 | -0.0086 +/- 0.0460 |
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=10

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0048 +/- 0.0055 | -0.0009 +/- 0.0015 |
| source reuse | 0.0040 +/- 0.0046 | -0.0001 +/- 0.0010 |
| population only | 0.0000 +/- 0.0000 | +0.0039 +/- 0.0047 |
| wrong-task source + relation | 0.0040 +/- 0.0047 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.0039 +/- 0.0047 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2202 +/- 0.3808 | +0.0001 +/- 0.0003 |
| source reuse | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0013 |
| population only | 0.2007 +/- 0.3917 | +0.0197 +/- 0.0244 |
| wrong-task source + relation | 0.2220 +/- 0.3804 | -0.0017 +/- 0.0030 |
| correct source + transport | 0.2203 +/- 0.3811 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7997 +/- 0.2702 | +0.0047 +/- 0.0071 |
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2632 +/- 0.0449 | -0.0006 +/- 0.0014 |
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 4.4171 +/- 0.7974 | -0.0073 +/- 0.0406 |
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=20

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0056 +/- 0.0065 | -0.0017 +/- 0.0021 |
| source reuse | 0.0040 +/- 0.0046 | -0.0001 +/- 0.0010 |
| population only | 0.0000 +/- 0.0000 | +0.0039 +/- 0.0047 |
| wrong-task source + relation | 0.0040 +/- 0.0047 | -0.0001 +/- 0.0003 |
| correct source + transport | 0.0039 +/- 0.0047 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2208 +/- 0.3806 | -0.0005 +/- 0.0009 |
| source reuse | 0.2210 +/- 0.3808 | -0.0007 +/- 0.0013 |
| population only | 0.2007 +/- 0.3917 | +0.0197 +/- 0.0244 |
| wrong-task source + relation | 0.2220 +/- 0.3804 | -0.0017 +/- 0.0030 |
| correct source + transport | 0.2203 +/- 0.3811 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7972 +/- 0.2689 | +0.0022 +/- 0.0106 |
| source reuse | 1.8099 +/- 0.2613 | +0.0149 +/- 0.0162 |
| population only | 2.2577 +/- 0.4548 | +0.4627 +/- 0.4693 |
| wrong-task source + relation | 1.7943 +/- 0.2765 | -0.0007 +/- 0.0027 |
| correct source + transport | 1.7950 +/- 0.2769 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.2630 +/- 0.0447 | -0.0008 +/- 0.0017 |
| source reuse | 0.2649 +/- 0.0446 | +0.0012 +/- 0.0027 |
| population only | 0.2527 +/- 0.0275 | -0.0111 +/- 0.0238 |
| wrong-task source + relation | 0.2637 +/- 0.0454 | -0.0000 +/- 0.0013 |
| correct source + transport | 0.2638 +/- 0.0461 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 4.3974 +/- 0.7895 | -0.0270 +/- 0.0576 |
| source reuse | 4.5031 +/- 0.7991 | +0.0787 +/- 0.0827 |
| population only | 5.4122 +/- 1.1998 | +0.9878 +/- 0.7085* |
| wrong-task source + relation | 4.4233 +/- 0.8218 | -0.0011 +/- 0.0391 |
| correct source + transport | 4.4244 +/- 0.8362 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
