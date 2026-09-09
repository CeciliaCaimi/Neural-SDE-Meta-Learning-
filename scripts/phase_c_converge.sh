#!/usr/bin/env bash
# Phase C: does the small-backbone advantage survive to convergence?
#
# Phase B showed the mechanism engages once the backbone is small enough:
# gain_vs_zero at 20k steps is 0.0005 (128ch), 0.144 (64ch), 0.363 (32ch).
# But every curve is still falling at 20k. The 128ch one fell 200-fold and hit
# zero; the others fell only 2-3 fold. Whether they level off or eventually reach
# zero decides whether the capacity finding is a real effect or a slower version
# of the same collapse, so both are run to 60k with k held at 32.
set -u
. "$(dirname "$0")/_common.sh"

run () {
  name=$1; shift
  echo "===== BEGIN $name ====="
  $PY -m runner.train \
    --steps 60000 --scheme domainshift --k 32 \
    --ckpt-every 30000 --diagnose-every 2000 \
    --run-name "$name" "$@" 2>&1 | grep -vE "$NOISE"
  echo "===== END $name (exit $?) ====="
}

run ds_cap32_60k --base-channels 32
run ds_cap64_60k --base-channels 64

echo "===== PHASE C COMPLETE ====="
