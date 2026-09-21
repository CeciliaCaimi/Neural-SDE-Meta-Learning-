# P1 pathology-family result

**Locked primary verdict: inconclusive: semantic evaluator failed the locked validity rule.**

The primary effect is correct-family minus wrong-family semantic consistency at K_T=0; positive favours correct-family transport.

Primary effect: +0.0000 +/- 0.0000 (95% CI half-width; n=6 outer-test families).

## Semantic-instrument validity

| Fold | Overall held-out accuracy | Minimum test-family accuracy | Pass |
|---:|---:|---:|:---:|
| 0 | 0.676 | 0.000 | no |
| 1 | 0.683 | 0.000 | no |
| 2 | 0.687 | 0.153 | no |

## K_T=0

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.0000 +/- 0.0000 | +0.0002 +/- 0.0004 |
| population only | 0.0000 +/- 0.0000 | +0.0002 +/- 0.0004 |
| wrong-task source + relation | 0.0000 +/- 0.0000 | +0.0002 +/- 0.0004 |
| correct source + transport | 0.0002 +/- 0.0004 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.1651 +/- 0.3227 | +0.0004 +/- 0.0005 |
| population only | 0.1641 +/- 0.3216 | +0.0015 +/- 0.0019 |
| wrong-task source + relation | 0.1656 +/- 0.3230 | +0.0000 +/- 0.0000 |
| correct source + transport | 0.1656 +/- 0.3230 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=1

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0052 +/- 0.0065 | +0.0026 +/- 0.0046 |
| source reuse | 0.0093 +/- 0.0118 | -0.0015 +/- 0.0021 |
| population only | 0.0000 +/- 0.0000 | +0.0078 +/- 0.0102 |
| wrong-task source + relation | 0.0074 +/- 0.0106 | +0.0004 +/- 0.0020 |
| correct source + transport | 0.0078 +/- 0.0102 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.1558 +/- 0.3054 | +0.0004 +/- 0.0005 |
| source reuse | 0.1567 +/- 0.3071 | -0.0004 +/- 0.0014 |
| population only | 0.1656 +/- 0.3245 | -0.0093 +/- 0.0188 |
| wrong-task source + relation | 0.1567 +/- 0.3066 | -0.0004 +/- 0.0009 |
| correct source + transport | 0.1563 +/- 0.3057 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7322 +/- 0.2193 | +0.0134 +/- 0.0227 |
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.3045 +/- 0.0194 | -0.0018 +/- 0.0037 |
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 5.7208 +/- 0.9739 | -0.0228 +/- 0.0745 |
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=2

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0076 +/- 0.0094 | +0.0002 +/- 0.0036 |
| source reuse | 0.0093 +/- 0.0118 | -0.0015 +/- 0.0021 |
| population only | 0.0000 +/- 0.0000 | +0.0078 +/- 0.0102 |
| wrong-task source + relation | 0.0074 +/- 0.0106 | +0.0004 +/- 0.0020 |
| correct source + transport | 0.0078 +/- 0.0102 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.1558 +/- 0.3054 | +0.0004 +/- 0.0005 |
| source reuse | 0.1567 +/- 0.3071 | -0.0004 +/- 0.0014 |
| population only | 0.1656 +/- 0.3245 | -0.0093 +/- 0.0188 |
| wrong-task source + relation | 0.1567 +/- 0.3066 | -0.0004 +/- 0.0009 |
| correct source + transport | 0.1563 +/- 0.3057 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7306 +/- 0.2201 | +0.0118 +/- 0.0127 |
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.3055 +/- 0.0191 | -0.0007 +/- 0.0012 |
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 5.7303 +/- 0.9850 | -0.0133 +/- 0.0315 |
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=5

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0082 +/- 0.0106 | -0.0004 +/- 0.0009 |
| source reuse | 0.0093 +/- 0.0118 | -0.0015 +/- 0.0021 |
| population only | 0.0000 +/- 0.0000 | +0.0078 +/- 0.0102 |
| wrong-task source + relation | 0.0074 +/- 0.0106 | +0.0004 +/- 0.0020 |
| correct source + transport | 0.0078 +/- 0.0102 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.1556 +/- 0.3050 | +0.0007 +/- 0.0009 |
| source reuse | 0.1567 +/- 0.3071 | -0.0004 +/- 0.0014 |
| population only | 0.1656 +/- 0.3245 | -0.0093 +/- 0.0188 |
| wrong-task source + relation | 0.1567 +/- 0.3066 | -0.0004 +/- 0.0009 |
| correct source + transport | 0.1563 +/- 0.3057 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7412 +/- 0.2117 | +0.0224 +/- 0.0119* |
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.3041 +/- 0.0190 | -0.0022 +/- 0.0018* |
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 5.6749 +/- 0.9871 | -0.0687 +/- 0.0379* |
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=10

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0089 +/- 0.0114 | -0.0011 +/- 0.0014 |
| source reuse | 0.0093 +/- 0.0118 | -0.0015 +/- 0.0021 |
| population only | 0.0000 +/- 0.0000 | +0.0078 +/- 0.0102 |
| wrong-task source + relation | 0.0074 +/- 0.0106 | +0.0004 +/- 0.0020 |
| correct source + transport | 0.0078 +/- 0.0102 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.1558 +/- 0.3054 | +0.0004 +/- 0.0005 |
| source reuse | 0.1567 +/- 0.3071 | -0.0004 +/- 0.0014 |
| population only | 0.1656 +/- 0.3245 | -0.0093 +/- 0.0188 |
| wrong-task source + relation | 0.1567 +/- 0.3066 | -0.0004 +/- 0.0009 |
| correct source + transport | 0.1563 +/- 0.3057 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7369 +/- 0.2147 | +0.0180 +/- 0.0080* |
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.3047 +/- 0.0204 | -0.0016 +/- 0.0014* |
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 5.7091 +/- 1.0350 | -0.0345 +/- 0.0374 |
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.

## K_T=20

### phenotype target share

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.0085 +/- 0.0107 | -0.0007 +/- 0.0013 |
| source reuse | 0.0093 +/- 0.0118 | -0.0015 +/- 0.0021 |
| population only | 0.0000 +/- 0.0000 | +0.0078 +/- 0.0102 |
| wrong-task source + relation | 0.0074 +/- 0.0106 | +0.0004 +/- 0.0020 |
| correct source + transport | 0.0078 +/- 0.0102 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### family semantic consistency

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.1554 +/- 0.3045 | +0.0009 +/- 0.0013 |
| source reuse | 0.1567 +/- 0.3071 | -0.0004 +/- 0.0014 |
| population only | 0.1656 +/- 0.3245 | -0.0093 +/- 0.0188 |
| wrong-task source + relation | 0.1567 +/- 0.3066 | -0.0004 +/- 0.0009 |
| correct source + transport | 0.1563 +/- 0.3057 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### phototype-statistic distance

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 1.7356 +/- 0.2158 | +0.0168 +/- 0.0092* |
| source reuse | 1.7452 +/- 0.2089 | +0.0263 +/- 0.0148* |
| population only | 3.0504 +/- 0.2209 | +1.3316 +/- 0.0892* |
| wrong-task source + relation | 1.7185 +/- 0.2170 | -0.0003 +/- 0.0016 |
| correct source + transport | 1.7188 +/- 0.2178 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### sliced Wasserstein

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 0.3039 +/- 0.0194 | -0.0024 +/- 0.0020* |
| source reuse | 0.3081 +/- 0.0192 | +0.0018 +/- 0.0022 |
| population only | 0.2683 +/- 0.0110 | -0.0379 +/- 0.0204* |
| wrong-task source + relation | 0.3064 +/- 0.0200 | +0.0001 +/- 0.0009 |
| correct source + transport | 0.3063 +/- 0.0199 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
### energy MMD

| Method | Raw mean +/- 95% CI | Correct-vs-method paired effect |
|---|---:|---:|
| target only | 5.6900 +/- 1.0112 | -0.0536 +/- 0.0371* |
| source reuse | 5.7946 +/- 1.0078 | +0.0510 +/- 0.0514 |
| population only | 6.8472 +/- 1.0089 | +1.1036 +/- 0.4970* |
| wrong-task source + relation | 5.7468 +/- 1.0114 | +0.0032 +/- 0.0193 |
| correct source + transport | 5.7436 +/- 1.0026 | -- |

Positive paired effects favour correct-family transport. A star means the paired 95% CI excludes zero.
