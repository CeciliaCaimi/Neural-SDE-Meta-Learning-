#!/usr/bin/env bash
# Preflight for Jing Peng's package. Run after setup.sh, and again after your edits.
#
#     bash handoff/Jing-Peng/verify.sh
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

# A fresh clone has no .venv, so _common.sh falls back to whatever python is on PATH.
# Say so in one line rather than emitting a wall of import tracebacks.
if ! $PY -c "import torch" >/dev/null 2>&1; then
    echo "FAIL: the interpreter in use ($PY) has no torch."
    echo "  This looks like a fresh clone. Bootstrap the environment first:"
    echo "      bash handoff/Jing-Peng/setup.sh"
    exit 1
fi

echo "== interpreter and device =="
$PY - <<'PYEOF'
import sys, torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("FAIL: no CUDA device. This package is 5.5 GPU-hours of training.")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
print(f"{n} device(s) -- the four training runs are independent and can go in parallel")
PYEOF

echo
echo "== dataset =="
if [ -z "${CIFAR100_ROOT:-}" ]; then
    echo "FAIL: CIFAR100_ROOT is not set. setup.sh printed the line to export."
    exit 1
fi
$PY - <<'PYEOF'
from domains.cifar100 import DEFAULT_ROOT, load_cifar100
raw = load_cifar100()
assert raw.images.shape == (60000, 32, 32, 3), raw.images.shape
print(f"CIFAR-100 at {DEFAULT_ROOT}: {raw.images.shape[0]} images")
PYEOF

echo
echo "== split file =="
$PY - <<'PYEOF'
import hashlib
path = "artifacts/cifar100_domainshift_c3.json"
want = "2ae9b81b639468fbbe5c6f087ed42c5d4eb924184f734988af00540768047708"
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
if got != want:
    raise SystemExit(f"FAIL: {path} is not the file the protocol card names\n"
                     f"  expected {want}\n  found    {got}")
from episodes.domainshift import load_domainshift
s = load_domainshift(path)
print(f"ok  {path}")
print(f"    relations {list(s.config.corruptions)}  severity {s.config.severity}")
print(f"    train/val/test fine classes "
      f"{s.n_episodes('train')}/{s.n_episodes('val')}/{s.n_episodes('test')}")
PYEOF

echo
echo "== the model_cls seam (your FiLM arm depends on it) =="
$PY - <<'PYEOF'
import inspect, torch
from config.base_config import BaseConfig
from models.score_model import ScoreModel
from training.loop import build, train
import models.unet  # noqa: F401

sig = inspect.signature(train)
assert "model_cls" in sig.parameters, "the seam is missing -- you are on the wrong commit"
cfg = BaseConfig(); cfg.model.k = 4
cfg.model.backbone_kwargs = dict(base_channels=32, channel_mult=(1, 2),
                                 num_res_blocks=1, attn_resolutions=())

class Probe(ScoreModel):
    pass

m, _, _ = build(cfg, torch.device("cpu"), Probe)
assert isinstance(m, Probe)
print("ok  build(cfg, device, model_cls) returns the subclass")
print("    write runner/train_film.py against this; do not edit training/loop.py")
PYEOF

echo
echo "== test suite =="
for t in test_score_identity test_backbone_swap test_gmm_analytic \
         test_cifar100_episodes test_document_conformance; do
    out=$($PY -m tests.$t 2>&1) || { echo "FAIL: tests.$t"; echo "$out" | tail -20; exit 1; }
    echo "ok  tests.$t  $(echo "$out" | grep -E 'passed' | tail -1)"
done

echo
echo "== throughput reference (RTX 5080, for comparison) =="
echo "    128 channels  11.5 it/s   ->  50 000 steps in  72 min"
echo "     64 channels  17.8 it/s"
echo "     32 channels  36.1 it/s   ->  60 000 steps in  28 min"
echo "    if your instance is much slower than this, say so in the results file"

echo
echo "preflight complete -- nothing above said FAIL"
