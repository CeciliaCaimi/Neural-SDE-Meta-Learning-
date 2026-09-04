#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
for k in 2 4 8; do
  echo "### k=$k steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k $k --decoder linear \
      --family unrelated --eval-tasks 4 > artifacts/stage1_k${k}_s80k.log 2>&1
done
$PY scripts/ft_reference.py \
  stage1_linear_k2_s80000_unrelated.pt stage1_linear_k4_s80000_unrelated.pt \
  stage1_linear_k8_s80000_unrelated.pt stage1_linear_k16_s80000_unrelated.pt \
  stage1_linear_k32_s80000_unrelated.pt stage1_linear_k64_s80000_unrelated.pt \
  > artifacts/ksweep_80k_full.txt 2>&1
echo "### done"
