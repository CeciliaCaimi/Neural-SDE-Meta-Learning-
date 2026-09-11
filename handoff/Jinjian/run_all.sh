#!/usr/bin/env bash
# The driver for Jinjian's package. Run it once the three new scripts exist; it will
# tell you which one is missing rather than failing obscurely.
#
#     bash handoff/Jinjian/run_all.sh
#
# Everything here is evaluation. Nothing trains, and nothing writes to artifacts/.
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

OUT=handoff/Jinjian/results
mkdir -p "$OUT"

C3=artifacts/cifar100_domainshift_c3.json
C1=artifacts/cifar100_domainshift.json

need() {
    if [ ! -f "$1" ]; then
        echo "not yet written: $1"
        echo "  see handoff/Jinjian/README.md for what it has to do"
        exit 1
    fi
}

need scripts/posthoc_controls.py
need evaluation/instruments.py
need scripts/gen_specificity.py

echo "===== E12a  post-hoc controls, three corruptions ====="
$PY scripts/posthoc_controls.py checkpoints/cifar_ds3_step50000.pt \
    --domainshift-path "$C3" --split val --n-episodes 24 --n-noise 4 \
    2>&1 | tee "$OUT/E12a_controls.txt"

echo
echo "===== E12a  post-hoc controls, single corruption ====="
$PY scripts/posthoc_controls.py checkpoints/cifar_ds_blur_step50000.pt \
    --domainshift-path "$C1" --split val --n-episodes 24 --n-noise 4 \
    2>&1 | tee -a "$OUT/E12a_controls.txt"

echo
echo "===== E12  generation-level task specificity ====="
# 20 validation classes x 4 coordinates x 256 samples, one shared column of initial noise
$PY scripts/gen_specificity.py checkpoints/cifar_ds3_step50000.pt \
    --domainshift-path "$C3" --split val --n-samples 256 \
    --grid-out "$OUT/gen_grid.png" \
    2>&1 | tee "$OUT/E12_generation.txt"

echo
echo "===== E14  sweep harness check (tool validation, not a result) ====="
bash scripts/phase_d_sweeps.sh checkpoints/ds_cap32_60k_step60000.pt \
    2>&1 | tee "$OUT/E14_harness_check.txt"

echo
echo "===== done ====="
echo "results in $OUT -- these are tracked by git; artifacts/ is not."
ls -la "$OUT"
