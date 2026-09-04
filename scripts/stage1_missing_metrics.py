"""Fill in the two quantities the document requires that stage 1 did not compute.

1. **transported-to-oracle coordinate error** (explicitly required by section 12.1)
   ||z_tilde_T - z*_oracle|| asks directly whether transport predicted the target coordinate,
   rather than inferring it from the resulting score error. The references are ||z_S - z*||
   (no transport) and ||z^enc_T - z*|| (sparse target only).

2. **Minimal structural diagnostics 1 and 2** (executive summary, section 2)
   1. does changing z materially change the score/noise field?  -> r_basis
   2. does the correct coordinate beat z=0 and a shuffled coordinate?
   Both were previously attached only to the CIFAR training loop and never run on stage 1.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from adaptation.budget import AdaptBudget                        # noqa: E402
from adaptation.coordinate import adapt                          # noqa: E402
from diffusion.forward import q_sample                           # noqa: E402
from diffusion.losses import denoising_loss                      # noqa: E402
from domains.gmm2d import build_related_task_family, build_task_family   # noqa: E402
from episodes.gmm_episodes import GMMEpisodeLoader               # noqa: E402
from ft_reference import load                                    # noqa: E402


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = sys.argv[1:] or ["stage1_linear_k16_s80000.pt"]
    n_tasks = 24

    for ck_name in names:
        cfg, model, enc, tr, ck = load(os.path.join(_ROOT, "checkpoints", ck_name), dev)
        if ck.get("family", "unrelated") == "related":
            fam = build_related_task_family(n_train=192, n_test=32, n_components=4,
                                            perturb=ck["perturb"], seed=ck["seed"], device=dev)
            label = f"related@{ck['perturb']}"
        else:
            fam = build_task_family(n_train=192, n_test=32, n_components=4,
                                    seed=ck["seed"], device=dev)
            label = "unrelated"
        loader = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=512,
                                  seed=ck["seed"] + 1)
        gen = torch.Generator(device=dev).manual_seed(0)

        print(f"\n{'='*78}\n{label}  k={ck['k']}  ({n_tasks} unseen tasks)\n{'='*78}")

        # ---- 1. Coordinate error ----
        acc = {"transported": [], "source_reuse": [], "target_only(K=5)": [], "zero": []}
        rb = {"correct": [], "zero": [], "shuffled": []}
        losses = {"correct": [], "zero": [], "shuffled": []}
        zs_all = []

        for ti in range(n_tasks):
            batch = loader.sample(k_shot=5, task_idx=ti)
            big = batch.task.target.sample(2048, gen).reshape(-1, 2, 1, 1)
            z_star = adapt("oracle", model, enc, tr, batch,
                           AdaptBudget(steps=500, lr=1e-2), oracle_data=big).z
            n = z_star.norm().clamp_min(1e-8)

            with torch.no_grad():
                z_s = enc(batch.src_support)
                rel = None if tr.relation_emb is None else batch.relation.reshape(1)
                z_tld = tr(z_s.unsqueeze(0), rel).squeeze(0)
                z_enc = enc(batch.tgt_support)

            acc["transported"].append(float((z_tld - z_star).norm() / n))
            acc["source_reuse"].append(float((z_s - z_star).norm() / n))
            acc["target_only(K=5)"].append(float((z_enc - z_star).norm() / n))
            acc["zero"].append(1.0)
            zs_all.append(z_tld)

        print("1. transported-to-oracle coordinate error  ||z - z*|| / ||z*||   (lower is better)")
        for k, v in acc.items():
            print(f"    {k:<20}{sum(v)/len(v):>8.4f}")

        # ---- 2. Minimal structural diagnostics ----
        z_stack = torch.stack(zs_all)
        for ti in range(n_tasks):
            batch = loader.sample(k_shot=5, task_idx=ti)
            nb = q_sample(model.schedule, batch.tgt_query)
            m = nb.x_t.shape[0]
            z_ok = zs_all[ti].unsqueeze(0).expand(m, -1)
            z_sh = zs_all[(ti + 1) % n_tasks].unsqueeze(0).expand(m, -1)   # another task's coordinate
            z_0 = torch.zeros_like(z_ok)
            with torch.no_grad():
                preds = model.eps_hat_many(nb.x_t, nb.t, [z_ok, z_0, z_sh])
                for nm, p in zip(("correct", "zero", "shuffled"), preds):
                    losses[nm].append(float(denoising_loss(nb.eps, p, nb.t, model.schedule)))
                rb["correct"].append(float(model.basis_usage(nb.x_t, nb.t, z_ok).mean()))

        mean = lambda d, k: sum(d[k]) / len(d[k])
        spread = torch.cdist(z_stack, z_stack)
        off = spread[~torch.eye(n_tasks, dtype=torch.bool, device=dev)]
        print("\n2. minimal structural diagnostics (executive summary, items 1 and 2)")
        print(f"    r_basis = ‖ε_res‖/‖ε_base‖        {mean(rb,'correct'):>8.4f}"
              f"   {'ok' if mean(rb,'correct') > 1e-3 else '<- basis barely used'}")
        print(f"    denoising loss, correct            {mean(losses,'correct'):>8.4f}")
        print(f"    denoising loss, z=0                {mean(losses,'zero'):>8.4f}"
              f"   gain {mean(losses,'zero')-mean(losses,'correct'):>+7.4f}")
        print(f"    denoising loss, shuffled           {mean(losses,'shuffled'):>8.4f}"
              f"   gain {mean(losses,'shuffled')-mean(losses,'correct'):>+7.4f}")
        print(f"    coordinate spread across tasks     "
              f"{float(off.mean()/z_stack.norm(dim=1).mean()):>8.4f}"
              f"   {'ok' if float(off.mean()/z_stack.norm(dim=1).mean()) > 0.05 else '<- collapsed'}")


if __name__ == "__main__":
    main()
