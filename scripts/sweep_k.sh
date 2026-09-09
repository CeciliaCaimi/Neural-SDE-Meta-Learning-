#!/usr/bin/env bash
# Phase A: choose the coordinate dimension k on the validation classes.
#
# Only k varies; everything else is held identical. Selection happens on validation
# because the protocol reserves the test classes for the final number.
set -u
. "$(dirname "$0")/_common.sh"

for k in 16 32 64; do
  echo "===== k=$k ====="
  $PY -m runner.train \
    --steps 20000 --scheme domainshift --k $k \
    --ckpt-every 20000 --diagnose-every 1000 \
    --run-name ds_k${k} 2>&1 | grep -vE "$NOISE"
done
echo "===== PHASE A COMPLETE ====="
