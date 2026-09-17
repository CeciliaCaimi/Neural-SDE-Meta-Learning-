#!/usr/bin/env bash
# The remaining GPU work, in order, on one device.
#
# Everything here contends for the same GPU, so it runs strictly sequentially and waits for
# A2's training to finish first. The wait is on a2_film_step50000.pt, which exists only when
# that run has completed its last step.
#
# Order and why:
#   1. Stage C training. Put ahead of A2's evaluation because the evaluation is read-only and
#      can wait, while nothing downstream of Stage C can start without the checkpoint.
#   2. A2 evaluation, both arms plus the cross-arm table.
#   3. C4 on the Stage C checkpoint. The 50k checkpoint is what runs here; the val diagnostics
#      in checkpoints/c_fitz_log.jsonl decide whether an earlier one should be preferred, and
#      if so C4 is re-run by hand. Selecting on val is what the val conditions are for.
#
#     bash scripts/stage_queue.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=handoff/results
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT"

echo "=== waiting for A2 training to finish  $(date) ==="
while [ ! -f checkpoints/a2_film_step50000.pt ]; do
  sleep 60
done
sleep 30                     # let the training process exit and free the device
echo "=== A2 training done  $(date) ==="

# ---- 1. Stage C training ---------------------------------------------------
# k=16 and the 128-channel backbone as stage A; the linear transport map, which A5 found
# indistinguishable from the residual MLP at nineteen times fewer parameters and which will
# behave better across five training tasks; M_S=16 passed explicitly, because A4 showed source
# evidence saturates below sixteen images and that is what makes this split constructible.
echo "=== Stage C training  $(date) ==="
$PY -m runner.train --scheme fitzpatrick --k 16 --steps 50000 \
    --transport-kind linear --m-source 16 \
    --diagnose-every 2000 --ckpt-every 5000 --run-name c_fitz \
    > "$OUT/C_train_fitz.log" 2>&1
echo "stage C training exit $?"

# ---- 2. A2 evaluation ------------------------------------------------------
echo "=== A2 evaluation  $(date) ==="
bash scripts/a2_eval.sh > "$OUT/A2_eval_driver.log" 2>&1
echo "a2 eval exit $?"

# ---- 3. C4 ----------------------------------------------------------------
echo "=== C4 scarcity sweep  $(date) ==="
$PY scripts/c4_scarcity.py checkpoints/c_fitz_step50000.pt \
    --split test --k-shots 1 2 3 5 8 12 20 --repeats 6 \
    --n-samples 96 --ddim-steps 50 --seed 4321 --progress 40 \
    --json-out "$OUT/C4_demographic_scarcity.json" \
    --grid-out "$OUT/C4_grid.png" \
    > "$OUT/C4_console.txt" 2> "$OUT/C4_progress.txt"
echo "c4 exit $?"

echo "=== all done  $(date) ==="
ls -la "$OUT"/A2_* "$OUT"/C4_* "$OUT"/C_train_fitz.log 2>/dev/null
