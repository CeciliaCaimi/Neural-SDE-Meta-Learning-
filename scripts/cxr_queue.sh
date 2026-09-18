#!/usr/bin/env bash
# Stage C on ChestX-ray14: wait for the download, then select capacity on val, as C4 taught.
#
# The Fitzpatrick run took stage A's configuration (128 channels, 50 000 steps) without asking
# whether the data funded it, overfitted by a factor of 2.4 on held-out conditions, trained the
# task coordinate out of the model (r_basis 0.447 -> 0.0024) and produced a C4 table that had
# to be discarded. That mistake is not repeated here: capacity is swept first and the
# checkpoint is chosen on the validation tasks.
#
# There is more data this time -- 14 310 images against 1 326 -- so 128 channels may well hold.
# The sweep is what decides, not the guess.
#
#     bash scripts/cxr_queue.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=handoff/results
export PYTHONIOENCODING=utf-8
export CHESTXRAY_ROOT="${CHESTXRAY_ROOT:-C:/T1D Meta Learning/dataset/chestxray14}"
mkdir -p "$OUT"

echo "=== waiting for the 45 GB fetch  $(date) ==="
while true; do
  if $PY -c "
import sys
from domains.chestxray import ZipIndex
try:
    idx = ZipIndex()          # raises unless all twelve archives are readable
    sys.exit(0 if len(idx) >= 112120 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    break
  fi
  sleep 120
done
echo "=== all twelve archives readable  $(date) ==="

# The split checks are cheap and they are what stands between a contaminated split and a
# meaningless result, so they run again here rather than being trusted from earlier.
$PY scripts/check_cxr_split.py > "$OUT/C_cxr_split_checks.txt" 2>&1
echo "split checks exit $?"

COMMON="--scheme chestxray --k 16 --transport-kind linear --m-source 16
        --diagnose-every 500 --ckpt-every 1000"

run () {   # name, channels, steps
  echo "=== $1: $2 channels, $3 steps  $(date) ==="
  $PY -m runner.train $COMMON --base-channels "$2" --steps "$3" --run-name "$1" \
      > "$OUT/C_cxr_select_$1.log" 2>&1
  echo "$1 exit $?"
}

run cxr128 128 20000
run cxr64   64 30000

echo "=== done  $(date) ==="
echo "Now read the validation curves and choose, as scripts/c_select_table.py does for"
echo "Fitzpatrick: best validation loss_correct, with r_basis and delta_task printed beside"
echo "it and NOT used to choose."
