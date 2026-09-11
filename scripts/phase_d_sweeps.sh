#!/usr/bin/env bash
# Phase D: the two evaluation-time sweeps the protocol asks for.
#
#     bash scripts/phase_d_sweeps.sh [checkpoint] [split-file]
#
# The checkpoint used to be written in here, which meant an argument was accepted and
# silently ignored -- a run that looks right and measured a different model. Pass it.
# The split file must be the one that checkpoint was trained on, or the corruption-count
# guard in cifar_table.py fires.
#
# All on the VALIDATION classes; test remains untouched.
set -u
. "$(dirname "$0")/_common.sh"

CK=${1:-checkpoints/ds_cap32_60k_step60000.pt}
DS=${2:-artifacts/cifar100_domainshift.json}

if [ ! -f "$CK" ]; then
    echo "no such checkpoint: $CK"
    exit 1
fi
echo "checkpoint : $CK"
echo "split file : $DS"
echo

echo "===== K_T sweep {1,2,5,10,20} at the training M_S ====="
$PY scripts/cifar_table.py "$CK" --split val --domainshift-path "$DS" \
  --n-episodes 32 --k-shots 1 2 5 10 20 --n-noise 4 2>&1 \
  | grep -vE "$NOISE"

echo
echo "===== M_S sweep {16,32,64,128,256,300} at K_T=1 ====="
for ms in 16 32 64 128 256 300; do
  echo "----- M_S = $ms -----"
  $PY scripts/cifar_table.py "$CK" --split val --domainshift-path "$DS" \
    --n-episodes 32 --k-shots 1 --n-noise 4 --m-source $ms 2>&1 \
    | grep -vE "$NOISE" | tail -16
done
echo "===== PHASE D COMPLETE ====="
