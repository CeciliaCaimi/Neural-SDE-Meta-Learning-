#!/usr/bin/env bash
# k=32 vs k=64 在**同为 80k 步**下比较,排除"容量大所以更难训"的混淆
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
for k in 32 64; do
  echo "### k=$k steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k $k --decoder linear \
      --family unrelated --eval-tasks 4 > artifacts/stage1_k${k}_s80k.log 2>&1
done
$PY scripts/ft_reference.py stage1_linear_k16_s80000_unrelated.pt \
    stage1_linear_k32_s80000_unrelated.pt stage1_linear_k64_s80000_unrelated.pt \
    > artifacts/ksweep_80k_reference.txt 2>&1
echo "### done"
