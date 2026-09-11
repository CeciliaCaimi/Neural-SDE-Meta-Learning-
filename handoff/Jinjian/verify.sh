#!/usr/bin/env bash
# Preflight for Jinjian's package. Run this before writing any code.
#
#     bash handoff/Jinjian/verify.sh
#
# It checks the five things that have actually gone wrong on this project: the wrong
# interpreter, a missing GPU, a split file that is not the one the protocol card names,
# a dataset that is not where the code looks, and a test suite that no longer passes.
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

# This package has no setup.sh because it runs in the checkout that already exists on
# this machine -- the one holding .venv, the dataset and the three checkpoints. If torch
# is missing, this is a fresh clone somewhere else, and none of those are here either.
if ! $PY -c "import torch" >/dev/null 2>&1; then
    echo "FAIL: the interpreter in use ($PY) has no torch."
    echo "  This package belongs in the existing checkout on the workstation, where the"
    echo "  virtual environment, CIFAR-100 and the three checkpoints already live."
    echo "  Update that checkout instead of cloning a new one:"
    echo "      git fetch origin"
    echo "      git checkout jinjian"
    exit 1
fi

echo "== interpreter and device =="
$PY - <<'PYEOF'
import sys, torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("FAIL: no CUDA device. This package samples 20480 images; it needs the GPU.")
print(f"device  {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
PYEOF

echo
echo "== dataset =="
$PY - <<'PYEOF'
from domains.cifar100 import DEFAULT_ROOT, load_cifar100
raw = load_cifar100()
assert raw.images.shape == (60000, 32, 32, 3), raw.images.shape
print(f"CIFAR-100 at {DEFAULT_ROOT}")
print(f"pooled {raw.images.shape[0]} images, {raw.images.dtype}")
PYEOF

echo
echo "== split files =="
$PY - <<'PYEOF'
import hashlib
expected = {
    "artifacts/cifar100_domainshift_c3.json":
        "2ae9b81b639468fbbe5c6f087ed42c5d4eb924184f734988af00540768047708",
    "artifacts/cifar100_domainshift.json":
        "bed53927daac3c490114af48c37b5325b9a2803a5fa1a89304bb177d7d5c5d74",
}
for path, want in expected.items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        raise SystemExit(f"FAIL: {path}\n  expected {want}\n  found    {got}")
    print(f"ok  {path}")
PYEOF

echo
echo "== checkpoints this package reads =="
$PY - <<'PYEOF'
import os, torch
rows = [("checkpoints/cifar_ds3_step50000.pt",     3,    "cifar100_domainshift_c3.json"),
        ("checkpoints/cifar_ds_blur_step50000.pt",  None, "cifar100_domainshift.json"),
        ("checkpoints/ds_cap32_60k_step60000.pt",   None, "cifar100_domainshift.json")]
missing = [p for p, _, _ in rows if not os.path.exists(p)]
if missing:
    raise SystemExit("FAIL: these checkpoints are missing and exist on no other machine:\n  "
                     + "\n  ".join(missing))
for path, want_rel, split in rows:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = sd["config"]["model"]
    assert m["n_relations"] == want_rel, (path, m["n_relations"], want_rel)
    print(f"ok  {os.path.basename(path):30s} step {sd['step']:>6}  "
          f"{m['backbone_kwargs']['base_channels']:>3} ch  k={m['k']:<3} -> pair with {split}")
PYEOF

echo
echo "== test suite =="
for t in test_score_identity test_backbone_swap test_gmm_analytic \
         test_cifar100_episodes test_document_conformance; do
    out=$($PY -m tests.$t 2>&1) || { echo "FAIL: tests.$t"; echo "$out" | tail -20; exit 1; }
    echo "ok  tests.$t  $(echo "$out" | grep -E 'passed' | tail -1)"
done

echo
echo "preflight complete -- nothing above said FAIL"
