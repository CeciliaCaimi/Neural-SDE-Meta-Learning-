#!/usr/bin/env bash
# One-time environment bootstrap on the AWS server.
#
#     bash handoff/Jing-Peng/setup.sh
#
# Creates the virtual environment, installs torch for the CUDA version present, fetches
# CIFAR-100 into a directory OUTSIDE the repository, and writes the environment line you
# need in every shell.
#
# CIFAR-100 is deliberately never committed to this repository, and the dataset directory
# on the machine where the results on record were produced also holds clinical data that
# must never reach GitHub. Keep the data outside the working tree.
set -eu
. "$(dirname "$0")/../../scripts/_common.sh"

DATA_DIR="${CIFAR100_DIR:-$HOME/data}"
CIFAR_ROOT="$DATA_DIR/cifar-100-python"

echo "== virtual environment =="
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "created .venv"
else
    echo ".venv already present"
fi
PY=.venv/bin/python

echo
echo "== torch =="
if ! $PY -c "import torch" 2>/dev/null; then
    echo "Installing torch. Match the index URL to the CUDA version on this instance:"
    echo "  nvidia-smi shows the driver's CUDA; cu128 wheels are what the results used."
    $PY -m pip install --quiet --upgrade pip
    $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY -m pip install --quiet -r requirements.txt
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo
echo "== CIFAR-100 =="
if [ -d "$CIFAR_ROOT" ]; then
    echo "already at $CIFAR_ROOT"
else
    mkdir -p "$DATA_DIR"
    echo "downloading into $DATA_DIR (169 MB)"
    curl -L -o "$DATA_DIR/cifar-100-python.tar.gz" \
        https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    tar -xzf "$DATA_DIR/cifar-100-python.tar.gz" -C "$DATA_DIR"
    rm -f "$DATA_DIR/cifar-100-python.tar.gz"
fi

export CIFAR100_ROOT="$CIFAR_ROOT"
$PY - <<'PYEOF'
from domains.cifar100 import DEFAULT_ROOT, load_cifar100
raw = load_cifar100()
assert raw.images.shape == (60000, 32, 32, 3), raw.images.shape
print(f"loaded {raw.images.shape[0]} images from {DEFAULT_ROOT}")
PYEOF

echo
echo "== put this in your shell profile, and in every shell you run training from =="
echo
echo "    export CIFAR100_ROOT=\"$CIFAR_ROOT\""
echo
echo "setup complete. Next: bash handoff/Jing-Peng/verify.sh"
