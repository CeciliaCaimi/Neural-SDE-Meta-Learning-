#!/bin/bash
# Phase D: the two evaluation-time sweeps the protocol asks for.
#
# Run on the 32-channel / 60k checkpoint, which is where the mechanism still has
# measurable life (gain_vs_zero 0.119, against 0.0005 for the 128-channel model).
# Running them on a collapsed checkpoint would measure nothing.
#
# All on the VALIDATION classes; test remains untouched.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
CK=checkpoints/ds_cap32_60k_step60000.pt

echo "===== K_T sweep {1,2,5,10,20} at the training M_S ====="
PYTHONIOENCODING=utf-8 $PY scripts/cifar_table.py "$CK" --split val \
  --n-episodes 32 --k-shots 1 2 5 10 20 --n-noise 4 2>&1 \
  | grep -vE "VisibleDeprecation|pickle.load"

echo
echo "===== M_S sweep {16,32,64,128,256,300} at K_T=1 ====="
for ms in 16 32 64 128 256 300; do
  echo "----- M_S = $ms -----"
  PYTHONIOENCODING=utf-8 $PY scripts/cifar_table.py "$CK" --split val \
    --n-episodes 32 --k-shots 1 --n-noise 4 --m-source $ms 2>&1 \
    | grep -vE "VisibleDeprecation|pickle.load" | tail -14
done
echo "===== PHASE D COMPLETE ====="
