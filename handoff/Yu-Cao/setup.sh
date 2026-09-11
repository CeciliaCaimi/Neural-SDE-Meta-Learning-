#!/usr/bin/env bash
# One-time environment bootstrap on macOS. CPU only; no dataset is needed.
#
#     bash handoff/Yu-Cao/setup.sh
set -eu
. "$(dirname "$0")/../../scripts/_common.sh"

echo "== virtual environment =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "created .venv"
else
    echo ".venv already present"
fi
PY=.venv/bin/python

echo
echo "== packages =="
$PY -m pip install --quiet --upgrade pip
# The default wheel is the right one on macOS: it carries CPU and MPS, no CUDA index needed.
$PY -m pip install --quiet torch
$PY -m pip install --quiet -r requirements.txt

$PY - <<'PYEOF'
import platform, torch
print(f"torch {torch.__version__} on {platform.machine()} / {platform.system()}")
print(f"threads {torch.get_num_threads()}")
print("cuda", torch.cuda.is_available(), "| mps",
      getattr(torch.backends, "mps", None) is not None
      and torch.backends.mps.is_available())
print("this package runs on the CPU by design -- see handoff/Yu-Cao/CLAUDE.md")
PYEOF

echo
echo "== no dataset required =="
echo "Stage 1 is synthetic: two-dimensional Gaussian mixtures generated from a seed."
echo "Do not download CIFAR-100 and do not set CIFAR100_ROOT."

echo
echo "setup complete. Next: bash handoff/Yu-Cao/verify.sh"
