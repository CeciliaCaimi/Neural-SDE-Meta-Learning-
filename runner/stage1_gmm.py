"""Stage 1: analytic validation on two-dimensional GMMs. Sections 3.1, 12.1, and step 2 of B.

    python -m runner.stage1_gmm [--steps N] [--k K]

This is the **first go/no-go**. The true score is analytic, so it answers directly:
  1. can s_0 + B(x,t) z represent the target score (is a small k enough)?
  2. does source-informed transport give a better starting point at small K_T than
     target-only, or than reusing z_S directly?
  3. do the generated point clouds converge to the true target distribution?

Failing here is far cheaper than failing on CIFAR, and cannot be blamed on an untrained backbone.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adaptation.budget import AdaptBudget                                  # noqa: E402
from adaptation.coordinate import adapt                                    # noqa: E402
from config.base_config import BaseConfig                                  # noqa: E402
from diffusion.forward import q_sample                                     # noqa: E402
from diffusion.losses import denoising_loss                                # noqa: E402
from diffusion.sampler import ddim_sample, make_eps_fn                     # noqa: E402
from diffusion.schedule import NoiseSchedule                               # noqa: E402
from domains.gmm2d import (                                                # noqa: E402
    RELATIONS, build_related_task_family, build_task_family,
)
from episodes.gmm_episodes import GMMEpisodeLoader, as_points              # noqa: E402
from evaluation.metrics_analytic import (                                  # noqa: E402
    energy_mmd, score_field_error, sliced_wasserstein,
)
from models.backbone import build_backbone                                 # noqa: E402
from models.score_model import ScoreModel                                  # noqa: E402
from models.set_encoder import VectorSetEncoder                            # noqa: E402
from models.transport import Transport                                     # noqa: E402
from training.meta_train import meta_step                                  # noqa: E402

import models.mlp_backbone  # noqa: F401,E402  -- triggers registration

RULE = "─" * 92
STRATEGIES = ("zero", "target_only", "source_reuse", "transport_no_refine", "transport",
              "oracle", "full_ft", "full_ft_oracle")


def build(cfg: BaseConfig, device, decoder: str = "linear"):
    sched = NoiseSchedule(cfg.diffusion.n_steps, cfg.diffusion.schedule)
    bb = build_backbone("mlp_vector", image_channels=2, hidden=256, depth=4,
                        feature_channels=128)
    model = ScoreModel(bb, sched, k=cfg.model.k, coord_decoder=decoder).to(device)
    enc = VectorSetEncoder(dim=2, feature_dim=128, k=cfg.model.k, hidden=128).to(device)
    tr = Transport(k=cfg.model.k, n_relations=len(RELATIONS),
                   relation_dim=cfg.model.relation_dim, hidden=cfg.model.transport_hidden).to(device)
    return model, enc, tr


def train(model, enc, tr, loader, cfg, steps: int) -> None:
    params = [p for m in (model, enc, tr) for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr)
    for m in (model, enc, tr):
        m.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = cfg.train.lr * min(1.0, step / 200)
        out = meta_step(model, enc, tr, loader.sample(), cfg)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == 1:
            m = out.metrics
            print(f"  step {step:>6}  L={m['L_meta']:.4f}  src={m['L_src']:.4f}  "
                  f"tgt={m['L_tgt']:.4f}  trans={m['L_trans']:.4f}  "
                  f"|dz|={m['|dz_transport|']:.3f}  {step/(time.time()-t0):.0f} it/s")
    for m in (model, enc, tr):
        m.eval()


@torch.no_grad()
def _denoise_loss(model, z, x0, cfg) -> float:
    nb = q_sample(model.schedule, x0)
    pred = model.eps_hat(nb.x_t, nb.t, z.unsqueeze(0).expand(x0.shape[0], -1))
    return float(denoising_loss(nb.eps, pred, nb.t, model.schedule,
                                cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma))


def evaluate(model, enc, tr, loader, cfg, k_shots, n_tasks: int, budget: AdaptBudget,
             n_gen: int = 2048) -> dict:
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(0)
    rows: dict[tuple[int, str], dict[str, list[float]]] = {}

    for k in k_shots:
        for ti in range(n_tasks):
            batch = loader.sample(k_shot=k, task_idx=ti)
            oracle_pts = batch.task.target.sample(1024, gen).reshape(-1, 2, 1, 1)
            for strat in STRATEGIES:
                b = budget.replace(steps=0) if strat == "transport_no_refine" else budget
                st = adapt(strat, model, enc, tr, batch, b,
                           cfg.diffusion.loss_weighting,
                           oracle_data=oracle_pts if strat in ("oracle", "full_ft_oracle") else None)
                z = st.z
                m_eval = st.model if st.model is not None else model   # full fine-tuning returns a modified copy

                sf = score_field_error(m_eval, z, batch.task.target, n_points=1024, generator=gen)
                dl = _denoise_loss(m_eval, z, batch.tgt_query, cfg)
                # clip_x0 to the real data range: means spread +-2.5, covariance <= 0.5, so 4 sigma is about +-4.5
                fake = as_points(ddim_sample(m_eval.schedule, make_eps_fn(m_eval, z),
                                             (n_gen, 2, 1, 1), dev, n_steps=50,
                                             generator=gen, clip_x0=8.0))
                real = batch.task.target.sample(n_gen, gen)
                sw = sliced_wasserstein(fake, real, generator=gen)
                mmd = energy_mmd(fake[:1024], real[:1024])

                d = rows.setdefault((k, strat), {})
                for name, val in (("score_rel_err", sf["score_rel_err"]), ("denoise", dl),
                                  ("SW2", sw), ("MMD", mmd),
                                  ("|z|", float(z.norm()))):
                    d.setdefault(name, []).append(val)
    return {kk: {n: sum(v) / len(v) for n, v in d.items()} for kk, d in rows.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-train-tasks", type=int, default=192)
    p.add_argument("--n-test-tasks", type=int, default=32)
    p.add_argument("--eval-tasks", type=int, default=12)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--decoder", choices=("linear", "nonlinear"), default="linear",
                   help="coordinate decoder h_eta; the matched comparison required by 3.1")
    p.add_argument("--family", choices=("unrelated", "related"), default="unrelated",
                   help="task family: mutually unrelated random GMMs, or a template plus perturbation")
    p.add_argument("--perturb", type=float, default=0.25,
                   help="perturbation strength of the related family; 0 = identical tasks, 1 = unrelated")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BaseConfig()
    cfg.model.k = a.k
    cfg.model.n_relations = len(RELATIONS)
    cfg.train.lr = 2e-3

    print(RULE)
    tag = a.family + (f"@{a.perturb}" if a.family == "related" else "")
    print(f"stage 1 | 2D GMM analytic validation   (k={a.k}, decoder {a.decoder}, family {tag}, device {dev})")
    print(RULE)

    if a.family == "related":
        fam = build_related_task_family(n_train=a.n_train_tasks, n_test=a.n_test_tasks,
                                        n_components=4, perturb=a.perturb,
                                        seed=a.seed, device=dev)
    else:
        fam = build_task_family(n_train=a.n_train_tasks, n_test=a.n_test_tasks,
                                n_components=4, seed=a.seed, device=dev)
    train_loader = GMMEpisodeLoader(fam["train"], dev, m_source=256, query_batch=256,
                                    k_shots=cfg.episodes.k_shots, seed=a.seed)
    test_loader = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=1024,
                                   k_shots=cfg.episodes.k_shots, seed=a.seed + 1)
    print(f"task family: train {len(fam['train'])} | test {len(fam['test'])} | "
          f"relations {list(RELATIONS)}\n")

    model, enc, tr = build(cfg, dev, a.decoder)
    n = sum(p_.numel() for m in (model, enc, tr) for p_ in m.parameters())
    print(f"parameters {n/1e6:.2f}M | {cfg.model.k} scalars optimised at deployment\n")

    print("training:")
    train(model, enc, tr, train_loader, cfg, a.steps)

    # Save the weights before evaluating, so a crash in evaluation does not cost the whole run
    os.makedirs(os.path.join(_ROOT, "checkpoints"), exist_ok=True)
    ck = os.path.join(_ROOT, "checkpoints",
                      f"stage1_{a.decoder}_k{a.k}_s{a.steps}_{a.family}{a.perturb if a.family=='related' else ''}.pt")
    torch.save({"model": model.state_dict(), "encoder": enc.state_dict(),
                "transport": tr.state_dict(), "k": a.k, "decoder": a.decoder,
                "steps": a.steps, "seed": a.seed,
                "family": a.family, "perturb": a.perturb}, ck)
    print(f"  weights saved: {os.path.basename(ck)}")

    print("\nevaluation (unseen GMM configurations at meta-test):")
    budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0)
    res = evaluate(model, enc, tr, test_loader, cfg, cfg.episodes.k_shots,
                   a.eval_tasks, budget)

    print()
    print(RULE)
    hdr = f"{'K_T':>4}  {'strategy':<20}{'score_rel_err':>15}{'denoise':>10}{'SW2':>9}{'MMD':>9}{'|z|':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in cfg.episodes.k_shots:
        for s in STRATEGIES:
            r = res[(k, s)]
            print(f"{k:>4}  {s:<20}{r['score_rel_err']:>15.4f}{r['denoise']:>10.4f}"
                  f"{r['SW2']:>9.4f}{r['MMD']:>9.4f}{r['|z|']:>8.3f}")
        print("-" * len(hdr))

    print("\ncriteria (the success conditions of section 3.1):")
    for k in cfg.episodes.k_shots:
        t_only = res[(k, "target_only")]["score_rel_err"]
        full = res[(k, "transport")]["score_rel_err"]
        reuse = res[(k, "source_reuse")]["score_rel_err"]
        orc = res[(k, "oracle")]["score_rel_err"]
        ft = res[(k, "full_ft_oracle")]["score_rel_err"]
        print(f"  K_T={k:>2}: transport vs target_only {t_only-full:+.4f} · "
              f"vs source_reuse {reuse-full:+.4f} | to oracle {full-orc:+.4f} | "
              f"oracle to full fine-tuning {orc-ft:+.4f}")


if __name__ == "__main__":
    main()
