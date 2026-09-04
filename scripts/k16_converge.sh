set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
# 文档默认 k=16,训到与 k=64 同样的 320k 步
$PY -m runner.stage1_gmm --steps 320000 --k 16 --decoder linear \
    --family unrelated --eval-tasks 4 > artifacts/stage1_k16_s320k.log 2>&1
$PY scripts/ft_reference.py \
  stage1_linear_k16_s80000_unrelated.pt stage1_linear_k16_s160000_unrelated.pt \
  stage1_linear_k16_s320000_unrelated.pt stage1_linear_k64_s320000_unrelated.pt \
  > artifacts/k16_convergence.txt 2>&1
echo "### done"
