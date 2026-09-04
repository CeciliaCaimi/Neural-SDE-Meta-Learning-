"""Supply the "how far can one get without the low-dimensional constraint" reference.

The earlier full_ft used the matched J of 25 steps, which barely moves a 1.3M-parameter net.
That is correct as a **fair baseline** (the document requires a matched budget) but useless
as an **upper-bound reference**. Here it is given a budget far beyond fairness, to ask:

    is the residual error of oracle (which optimises only k numbers) the ceiling of the

Nothing is trained here; checkpoints are read directly.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from adaptation.budget import AdaptBudget                        # noqa: E402
from adaptation.coordinate import adapt                          # noqa: E402
from config.base_config import BaseConfig                        # noqa: E402
from domains.gmm2d import (                                      # noqa: E402
    RELATIONS, build_related_task_family, build_task_family,
)
from episodes.gmm_episodes import GMMEpisodeLoader               # noqa: E402
from evaluation.metrics_analytic import score_field_error        # noqa: E402
from runner.stage1_gmm import build                              # noqa: E402

RULE = "─" * 74


def load(ck_path: str, dev):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = BaseConfig()
    cfg.model.k = ck["k"]
    cfg.model.n_relations = len(RELATIONS)
    model, enc, tr = build(cfg, dev, ck["decoder"])
    model.load_state_dict(ck["model"]); enc.load_state_dict(ck["encoder"])
    tr.load_state_dict(ck["transport"])
    for m in (model, enc, tr):
        m.eval()
    return cfg, model, enc, tr, ck


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks, k_shot = 6, 20
    names = sys.argv[1:] or ["stage1_linear_k16_s80000.pt"]

    print(f"{'checkpoint':<30}{'zero':>8}{'oracle':>9}{'bound':>9}{'budget grid span':>18}{'captured':>10}")
    print("=" * 88)
    for ck_name in names:
        path = os.path.join(_ROOT, "checkpoints", ck_name)
        cfg, model, enc, tr, ck = load(path, dev)
        if ck.get("family", "unrelated") == "related":
            fam = build_related_task_family(n_train=192, n_test=32, n_components=4,
                                            perturb=ck["perturb"], seed=ck["seed"], device=dev)
        else:
            fam = build_task_family(n_train=192, n_test=32, n_components=4,
                                    seed=ck["seed"], device=dev)
        loader = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=1024,
                                  k_shots=cfg.episodes.k_shots, seed=ck["seed"] + 1)
        gen = torch.Generator(device=dev).manual_seed(0)

        def run(strat, budget):
            errs = []
            for ti in range(n_tasks):
                batch = loader.sample(k_shot=k_shot, task_idx=ti)
                big = batch.task.target.sample(2048, gen).reshape(-1, 2, 1, 1)
                st = adapt(strat, model, enc, tr, batch, budget,
                           cfg.diffusion.loss_weighting, oracle_data=big)
                m_eval = st.model if st.model is not None else model
                errs.append(score_field_error(m_eval, st.z, batch.task.target,
                                              n_points=1024, generator=gen)["score_rel_err"])
            return sum(errs) / len(errs)

        e_zero = run("zero", AdaptBudget(steps=0))
        # oracle is verified to converge by J ~ 500 (J=8000 and 3x the learning rate do not improve)
        e_orc = run("oracle", AdaptBudget(steps=500, lr=1e-2))

        # Upper-bound reference: full fine-tuning is **non-monotone** in the budget, overfitting
        # the 2048 target samples while score_rel_err measures against the **true** score, so
        # overfitting raises the error. The minimum over a budget grid is the sound estimate.
        ft_grid = [(500, 3e-4), (2000, 3e-4), (8000, 3e-4), (8000, 1e-4)]
        ft_vals = [run("full_ft_oracle", AdaptBudget(steps=j, lr_weights=lr))
                   for j, lr in ft_grid]
        e_ft = min(ft_vals)
        frac = (e_zero - e_orc) / max(1e-9, e_zero - e_ft)
        label = ck.get("family","unrelated")
        if label == "related": label += f"@{ck['perturb']}"
        label = f"k={ck['k']} {ck['steps']//1000}k {label}"
        spread = f"[{min(ft_vals):.3f},{max(ft_vals):.3f}]"
        print(f"{label:<30}{e_zero:>8.4f}{e_orc:>9.4f}{e_ft:>9.4f}{spread:>16}{frac*100:>10.1f}%")


if __name__ == "__main__":
    main()
