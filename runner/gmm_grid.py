"""Sharded driver for the GMM boundary phase diagram (main alpha x eta grid, K_T=0).

Runs cells whose global index % nshards == shard, so N copies (one per GPU) cover the grid.
Writes one JSON object per cell (with per-task rows) to --out, flushed per cell.

  CUDA_VISIBLE_DEVICES=0 python -m runner.gmm_grid --shard 0 --nshards 4 --seeds 5 --out rows0.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig
from runner.gmm_boundary import ALPHAS, ETAS, run_cell


def all_cells(seeds):
    cells = []
    for seed in range(seeds):
        for a in ALPHAS:
            for e in ETAS:
                cells.append({"alpha": a, "eta": e, "seed": seed, "kt": 0,
                              "transport": "linear", "ms": 32})
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg = BaseConfig(); cfg.model.k = 16; cfg.model.n_relations = None; cfg.train.lr = 2e-3

    cells = all_cells(a.seeds)
    mine = [c for i, c in enumerate(cells) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}/{a.nshards}] device={dev} cells={len(mine)}/{len(cells)} steps={a.steps}",
          flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for n, c in enumerate(mine):
            t0 = time.time()
            r = run_cell(c["alpha"], c["eta"], c["seed"], a.steps, dev, cfg,
                         kt=c["kt"], transport_kind=c["transport"], ms=c["ms"])
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[shard {a.shard}] {n+1}/{len(mine)} a={c['alpha']} e={c['eta']} "
                  f"seed={c['seed']} G_source={r['G_source_mean']:+.4f} "
                  f"bayes_id={r['diagnostics']['bayes_id']:.3f} "
                  f"learned_id={r['diagnostics']['learned_id']:.3f} "
                  f"oracle_r2={r['diagnostics']['oracle_r2']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"[shard {a.shard}] DONE {len(mine)} cells", flush=True)


if __name__ == "__main__":
    main()
