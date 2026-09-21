# Remaining blockers

1. No validated phenotype classifier/calibration exists for generated Fitzpatrick images;
   the primary saved hard-label target-share metric is not target-sensitive on real images.
2. The family/disease semantic instruments fail their fixed per-task validity requirements.
3. Fitzpatrick has only six P1 and five P4 independent families, and no patient identifier.
4. P4 differs from P1 in eligible-family composition as well as target phototypes.
5. CIFAR representation and task-scarcity comparisons use one training seed per arm.
6. Matched latency, FLOPs and peak-memory measurements are unavailable.
7. Protocol chronology is supported by local timestamps, not an external immutable registry.

These blockers limit claims; none justifies reopening P6 or altering the frozen method in this
revision.
