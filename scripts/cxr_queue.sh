#!/usr/bin/env bash
# Stage C on ChestX-ray14: check the split, then select capacity on the validation tasks.
#
# The Fitzpatrick run took stage A's configuration (128 channels, 50 000 steps) without asking
# whether the data funded it, overfitted by a factor of 2.4 on held-out conditions, trained the
# task coordinate out of the model (r_basis 0.447 -> 0.0024) and produced a C4 table that had
# to be discarded. So capacity is swept first and the checkpoint is chosen on validation tasks.
#
# Split v2 since the audit of 2026-09-18 (see runner/build_cxr_splits.py): source 45-60, the
# richest bin; M_S = 64 drawn per episode from a source pool; whole-patient allocation, so the
# K_T support and the held-out query of a (task, relation) never share a chest. The first
# sweep (cxr128) ran on v1 and is not used for anything.
#
#     bash scripts/cxr_queue.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=handoff/results
SPLIT=artifacts/chestxray_split_v2.json
export PYTHONIOENCODING=utf-8
export CHESTXRAY_ROOT="${CHESTXRAY_ROOT:-C:/T1D Meta Learning/dataset/chestxray14}"
mkdir -p "$OUT"

# The split checks stand between a contaminated split and a meaningless result. The first
# version of this script printed their exit code and trained anyway; now a failure stops it.
if ! $PY scripts/check_cxr_split.py "$SPLIT" > "$OUT/C_cxr_split_checks_v2.txt" 2>&1; then
  echo "split checks FAILED -- see $OUT/C_cxr_split_checks_v2.txt; nothing was trained"
  exit 1
fi
echo "split checks passed  $(date)"

COMMON="--scheme chestxray --cxr-path $SPLIT --k 16 --transport-kind linear --m-source 64
        --diagnose-every 500 --ckpt-every 1000"

run () {   # name, channels, steps
  echo "=== $1: $2 channels, $3 steps  $(date) ==="
  $PY -m runner.train $COMMON --base-channels "$2" --steps "$3" --run-name "$1" \
      > "$OUT/C_cxr_select_$1.log" 2>&1
  echo "$1 exit $?"
}

run cxr2_128 128 20000
run cxr2_64   64 30000

echo "=== done  $(date) ==="
echo "Choose on validation loss_correct, with r_basis and delta_task printed beside it and"
echo "NOT used to choose -- as scripts/c_select_table.py does for Fitzpatrick."
