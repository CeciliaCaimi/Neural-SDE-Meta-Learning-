# P3 source-only task separability

Status: **completed; the proposed positive association is not supported.**

Official ImageNet ResNet-18 features were computed only from Fitzpatrick I--III images.
Per-image 512-dimensional penultimate features were L2-normalised and averaged by disease;
separation is Euclidean distance to the preassigned different-family wrong source. These
scores were joined to the K_T=0 correct-minus-wrong semantic effect from P2.

Across four independent test diseases, Spearman rho was -0.775 with a task-bootstrap interval
reported as [-1.000, -0.775]. The small-sample bootstrap is highly discrete: only 6,793 of
10,000 resamples had defined rank variance. More importantly, three of four semantic effects
were exactly zero under an evaluator that scored 0--12.5% on three test diseases.

Conclusion: greater source-task separability did not predict a greater correct-source
advantage in the locked analysis. Because the outcome instrument is weak and n=4, this is
best described as unsupported/unresolved, not evidence for a negative causal mechanism. The
paper should not use task separability as its central explanation for the Fitzpatrick result.
