#!/usr/bin/env bash
# Phase B: is the basis unused because the backbone absorbs the domain difference?
#
# The k sweep settled that coordinate capacity is not the bottleneck: k = 16, 32 and
# 64 all decay to gain_vs_zero < 0.001, and a larger k only raises the early peak.
# That pattern is what a backbone with enough capacity to model every task jointly
# would produce, so the next variable to move is the backbone, not the coordinate.
#
# k is held at 32 throughout. The 128-channel point already exists as ds_k32.
#   cap64  : same architecture, narrower  -- a clean width-only comparison with 128
#   cap32  : narrower and shallower (one fewer level, no attention) -- a larger step,
#            so read it as a wider capacity range rather than as pure width
#
# If the basis engages at low capacity, the fault was backbone size and the method
# stands. If it stays at zero, the finding is about natural images themselves.
set -u
. "$(dirname "$0")/_common.sh"

run () {
  name=$1; shift
  echo "===== BEGIN $name ====="
  $PY -m runner.train \
    --steps 20000 --scheme domainshift --k 32 \
    --ckpt-every 20000 --diagnose-every 1000 \
    --run-name "$name" "$@" 2>&1 | grep -vE "$NOISE"
  echo "===== END $name (exit $?) ====="
}

run ds_cap64 --base-channels 64
run ds_cap32 --base-channels 32

echo "===== PHASE B COMPLETE ====="
