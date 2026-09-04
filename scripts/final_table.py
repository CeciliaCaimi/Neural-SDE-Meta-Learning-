"""Produce the main baseline table for the report, using the current (corrected) code.

Differences from the earlier tables:
  - zero now means true no-adaptation (no refinement); it previously refined from 0
  - full_ft reports the minimum over a budget grid rather than one fixed budget
"""
from __future__ import annotations
import os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from adaptation.budget import AdaptBudget
from adaptation.coordinate import adapt
from diffusion.forward import q_sample
from diffusion.losses import denoising_loss
from diffusion.sampler import ddim_sample, make_eps_fn
from domains.gmm2d import build_task_family
from episodes.gmm_episodes import GMMEpisodeLoader, as_points
from evaluation.metrics_analytic import score_field_error, sliced_wasserstein
from ft_reference import load

ROWS = [("no adaptation (z=0)","zero"), ("target-only","target_only"),
        ("source reuse (z_S)","source_reuse"), ("transport, no refine","transport_no_refine"),
        ("transport (ours)","transport"),
        ("full fine-tuning, J=25","full_ft"),        # matched budget
        ("full fine-tuning, J=2000","full_ft_long"),  # generous budget, to expose overfitting
        ("abundant-target oracle","oracle")]

def main():
    dev = torch.device("cuda")
    cfg, model, enc, tr, ck = load(os.path.join(_ROOT,"checkpoints",
                                  "stage1_linear_k16_s80000_unrelated.pt"), dev)
    fam = build_task_family(n_train=192, n_test=32, n_components=4, seed=ck["seed"], device=dev)
    ld = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=1024, seed=ck["seed"]+1)
    gen = torch.Generator(device=dev).manual_seed(0)
    n_tasks = 12
    bud = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0)
    ft_bud = AdaptBudget(steps=2000, lr_weights=3e-4)

    print(f"k={ck['k']} | {ck['steps']//1000}k steps | {n_tasks} unseen tasks | 2D Gaussian mixtures")
    print(f"{'method':<24}" + "".join(f"{'K='+str(k):>22}" for k in (1,5,20)))
    print(f"{'':24}" + "".join(f"{'score err':>11}{'SW2':>11}" for _ in (1,5,20)))
    print("-"*90)
    for label, strat in ROWS:
        cells = []
        for kt in (1,5,20):
            se, sw = [], []
            for ti in range(n_tasks):
                b = ld.sample(k_shot=kt, task_idx=ti)
                big = b.task.target.sample(2048, gen).reshape(-1,2,1,1)
                real_strat = "full_ft" if strat == "full_ft_long" else strat
                if strat == "full_ft_long":   use = ft_bud
                elif strat == "full_ft":      use = bud.replace(lr_weights=3e-4)
                else:                          use = bud
                st = adapt(real_strat, model, enc, tr, b, use, "simple", oracle_data=big)
                me = st.model if st.model is not None else model
                se.append(score_field_error(me, st.z, b.task.target, n_points=1024,
                                            generator=gen)["score_rel_err"])
                fake = as_points(ddim_sample(me.schedule, make_eps_fn(me, st.z),
                                 (1024,2,1,1), dev, n_steps=50, generator=gen, clip_x0=8.0))
                sw.append(sliced_wasserstein(fake, b.task.target.sample(1024, gen), generator=gen))
            cells += [sum(se)/len(se), sum(sw)/len(sw)]
        print(f"{label:<24}" + "".join(f"{v:>11.4f}" for v in cells))

if __name__ == "__main__":
    main()
