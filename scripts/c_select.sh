#!/usr/bin/env bash
# Stage C model selection on the validation conditions.
#
# c_fitz at 50 000 steps overfits badly: the training target loss falls 0.0316 -> 0.0085
# while the validation loss on the three held-out conditions rises 0.0628 -> 0.1527, and
# r_basis collapses 0.447 -> 0.0024, i.e. the task coordinate is trained out of existence.
# The training side draws from 808 images behind five conditions, so 50 000 steps is about
# two thousand epochs; scaling stage A's budget by images gives roughly 1 100.
#
# Capacity is the lever that is already exposed, so this sweeps it against budget with dense
# checkpoints and selects on val -- which is what the validation conditions are for. Note
# --base-channels at or below 48 also drops one downsample level, attention and one residual
# block per stage, so 32 is a much smaller network and not only a narrower one.
#
# Two quantities decide, not one. The best validation loss is useless if r_basis has
# collapsed: that model is the unconditioned backbone and every arm of C4 would read the same.
# The selection is made by hand from the printed table.
#
#     bash scripts/c_select.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=handoff/results
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"

COMMON="--scheme fitzpatrick --k 16 --transport-kind linear --m-source 16
        --diagnose-every 500 --ckpt-every 1000"

run () {                      # name, channels, steps
  echo "=== $1: $2 channels, $3 steps  $(date) ==="
  $PY -m runner.train $COMMON --base-channels "$2" --steps "$3" --run-name "$1" \
      > "$OUT/C_select_$1.log" 2>&1
  echo "$1 exit $?"
}

run cs128 128  8000
run cs64   64 20000
run cs32   32 30000

echo "=== done  $(date) ==="
