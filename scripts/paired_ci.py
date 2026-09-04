"""Paired comparison with confidence intervals. Section 13.3 requires results "with confidence
intervals across held-out tasks and random seeds", which the earlier tables lacked entirely.

Method: run two strategies on the same task and the same target samples, take the **per-task
difference**, then average and form a 95% CI. Task variance cancels; only the method gap remains.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from adaptation.budget import AdaptBudget                        # noqa: E402
from adaptation.coordinate import adapt                          # noqa: E402
from domains.gmm2d import build_related_task_family, build_task_family   # noqa: E402
from episodes.gmm_episodes import GMMEpisodeLoader               # noqa: E402
from evaluation.metrics_analytic import score_field_error        # noqa: E402
from ft_reference import load                                    # noqa: E402

PAIRS = [("transport", "target_only"), ("transport", "source_reuse"),
         ("transport", "transport_no_refine"), ("transport", "zero")]


def ci95(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_names = sys.argv[1:] or ["stage1_linear_k16_s80000.pt"]
    n_tasks, n_seeds = 32, 3
    budget = AdaptBudget(steps=25, lr=1e-2, beta0=1.0)

    for ck_name in ck_names:
        cfg, model, enc, tr, ck = load(os.path.join(_ROOT, "checkpoints", ck_name), dev)
        if ck.get("family", "unrelated") == "related":
            fam = build_related_task_family(n_train=192, n_test=32, n_components=4,
                                            perturb=ck["perturb"], seed=ck["seed"], device=dev)
            label = f"related@{ck['perturb']}"
        else:
            fam = build_task_family(n_train=192, n_test=32, n_components=4,
                                    seed=ck["seed"], device=dev)
            label = "unrelated"

        print(f"\n{'='*82}\n{ck_name}   ({label}, k={ck['k']})   "
              f"{n_tasks} unseen tasks x {n_seeds} seeds, paired difference +- 95% CI\n{'='*82}")
        print(f"  {'comparison':<34}" + "".join(f"{'K_T='+str(k):>17}" for k in (1, 5, 20)))
        print("  " + "-" * 85)

        for a, b in PAIRS:
            cells = []
            for kt in (1, 5, 20):
                diffs = []
                for seed in range(n_seeds):
                    loader = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=512,
                                              seed=1000 * seed + ck["seed"])
                    gen = torch.Generator(device=dev).manual_seed(seed)
                    for ti in range(n_tasks):
                        batch = loader.sample(k_shot=kt, task_idx=ti)
                        errs = {}
                        for strat in (a, b):
                            st = adapt(strat, model, enc, tr, batch, budget,
                                       cfg.diffusion.loss_weighting)
                            errs[strat] = score_field_error(model, st.z, batch.task.target,
                                                            n_points=512, generator=gen)["score_rel_err"]
                        diffs.append(errs[b] - errs[a])       # > 0 means a is better
                m, h = ci95(diffs)
                sig = "*" if abs(m) > h else " "
                cells.append(f"{m:>+7.4f}±{h:<6.4f}{sig}")
            print(f"  {a + ' vs ' + b:<34}" + "".join(f"{c:>17}" for c in cells))
        print("  " + "-" * 85)
        print("  * = the 95% CI excludes 0 (difference resolvable); positive favours the former")


if __name__ == "__main__":
    main()
