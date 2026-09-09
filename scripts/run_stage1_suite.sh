#!/usr/bin/env bash
# Stage 1: the three follow-up experiments the source document asks for in
# sections 3.1 and 12.1.
#
#   1. linear against non-linear coordinate decoder, k = 16, matched at 80k steps
#   2. the k sweep, linear decoder, a uniform 40k steps
set -u
. "$(dirname "$0")/_common.sh"

for dec in linear nonlinear; do
  echo "### decoder=$dec k=16 steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder $dec --eval-tasks 8 \
      > artifacts/stage1_dec-${dec}_k16.log 2>&1
done

# 2) the k sweep, linear decoder, a uniform 40k steps
for k in 2 4 8 16 32; do
  echo "### k=$k steps=40000"
  $PY -m runner.stage1_gmm --steps 40000 --k $k --decoder linear --eval-tasks 6 \
      > artifacts/stage1_ksweep_k${k}.log 2>&1
done
echo "### done"
