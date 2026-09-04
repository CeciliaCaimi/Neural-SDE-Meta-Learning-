#!/usr/bin/env bash
# 阶段 1 的三组后续实验（文档 §3.1 / §12.1 要求的对照）
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
mkdir -p artifacts

# 1) 线性 vs 非线性坐标解码器，k=16，同为 80k 步（matched）
for dec in linear nonlinear; do
  echo "### decoder=$dec k=16 steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder $dec --eval-tasks 8 \
      > artifacts/stage1_dec-${dec}_k16.log 2>&1
done

# 2) k 扫描，线性解码器，统一 40k 步
for k in 2 4 8 16 32; do
  echo "### k=$k steps=40000"
  $PY -m runner.stage1_gmm --steps 40000 --k $k --decoder linear --eval-tasks 6 \
      > artifacts/stage1_ksweep_k${k}.log 2>&1
done
echo "### done"
