#!/bin/bash
# Phase A: 在 val 类上选坐标维度 k。只有 k 变化，其余全部相同。
cd "$(dirname "$0")/.."
for k in 16 32 64; do
  echo "===== k=$k ====="
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m runner.train \
    --steps 20000 --scheme domainshift --k $k \
    --ckpt-every 20000 --diagnose-every 1000 \
    --run-name ds_k${k} 2>&1 | grep -vE "VisibleDeprecation|pickle.load"
done
echo "===== Phase A 完成 ====="
