#!/usr/bin/env bash
# Preflight for Yu Cao's package. Run after setup.sh.
#
#     bash handoff/Yu-Cao/verify.sh
#
# It ends with a throughput measurement, so you know before you start how long the
# relatedness sweep will take on this particular Mac.
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

echo "== interpreter =="
$PY - <<'PYEOF'
import platform, sys, torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}")
print(f"{platform.system()} / {platform.machine()}  |  {torch.get_num_threads()} CPU threads")
if torch.cuda.is_available():
    print("note: a CUDA device is visible; this package is still CPU-only by design")
PYEOF

echo
echo "== analytic ground truth =="
out=$($PY -m tests.test_gmm_analytic 2>&1) || { echo "FAIL"; echo "$out" | tail -20; exit 1; }
echo "ok  tests.test_gmm_analytic  $(echo "$out" | grep passed | tail -1)"
echo "    (this is the test that checks the true score two independent ways --"
echo "     everything you measure rests on it)"

echo
echo "== the rest of the suite =="
for t in test_score_identity test_backbone_swap test_cifar100_episodes \
         test_document_conformance; do
    out=$($PY -m tests.$t 2>&1) || { echo "FAIL: tests.$t"; echo "$out" | tail -20; exit 1; }
    echo "ok  tests.$t  $(echo "$out" | grep -E 'passed' | tail -1)"
done

echo
echo "== stage-1 throughput on this machine =="
$PY - <<'PYEOF'
import time, torch
from config.base_config import BaseConfig
from domains.gmm2d import build_task_family
from episodes.gmm_episodes import GMMEpisodeLoader
from runner.stage1_gmm import build
from training.meta_train import meta_step

cfg = BaseConfig(); cfg.model.k = 16
dev = torch.device("cpu"); torch.manual_seed(12345)
fam = build_task_family(n_train=192, n_test=32, n_components=4, seed=12345, device=dev)
loader = GMMEpisodeLoader(fam["train"], dev, m_source=256, query_batch=256,
                          k_shots=cfg.episodes.k_shots, seed=12345)
model, enc, tr = build(cfg, dev)
params = [p for m in (model, enc, tr) for p in m.parameters()]
opt = torch.optim.AdamW(params, lr=2e-4)
for _ in range(10):                                  # warm up
    out = meta_step(model, enc, tr, loader.sample(), cfg)
    opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
t0 = time.time(); N = 100
for _ in range(N):
    out = meta_step(model, enc, tr, loader.sample(), cfg)
    opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
ips = N / (time.time() - t0)
print(f"{ips:.0f} it/s   (54 it/s on the reference machine's CPU)")
print(f"  one 80000-step run          {80000/ips/60:>5.0f} min")
print(f"  the five-point sweep at 80k {5*80000/ips/3600:>5.1f} h")
print(f"  the five-point sweep at 40k {5*40000/ips/3600:>5.1f} h")
if ips < 20:
    print("  slow: use 40000 steps for the sweep, and say so in the results file")
PYEOF

echo
echo "preflight complete -- nothing above said FAIL"
