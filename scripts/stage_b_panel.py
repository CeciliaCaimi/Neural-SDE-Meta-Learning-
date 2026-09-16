"""Stage B: the cheap positive control on two-dimensional Gaussian mixtures.

    python scripts/stage_b_panel.py --ckpt checkpoints/stage1_linear_k16_s80000_unrelated.pt
    python scripts/stage_b_panel.py --sweep --out handoff/Yu-Cao/results/B_panel.txt

Stage 1 is where the ground truth is analytic and the mechanism is known to work, so it
calibrates the readings that CIFAR gives. Three things, following the plan:

  1. the four-way panel -- own, mean, shuffled and zero coordinate;
  2. the coordinate-loss profile along z_bar + s (z_own - z_bar) for s in {0, .5, 1, 1.5, 2},
     where a clear minimum at s = 1 is expected. **On images that profile is flat, and the
     contrast between the two is the point of running this at all**;
  3. the relatedness sweep, over the checkpoints that already exist.

Two traps this script is written around.

**Average across tasks, not across the batch.** Every query point inside one task shares
one coordinate, so averaging along the batch dimension says nothing about task
specificity; it is the easiest way to get a confident wrong answer here.

**Compare the coordinates on identical noise.** q_sample draws (t, eps) from the global
stream unless it is handed a generator, and four coordinates evaluated on four different
noise draws differ by the noise rather than by the coordinate. Every panel entry below
shares one seeded stream per task, and the differences are paired per task.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                                  # noqa: E402
from diffusion.forward import q_sample                                     # noqa: E402
from diffusion.losses import denoising_loss                                # noqa: E402
from domains.gmm2d import (                                                # noqa: E402
    RELATIONS, build_related_task_family, build_task_family,
)
from episodes.gmm_episodes import GMMEpisodeLoader                         # noqa: E402
from runner.stage1_gmm import build                                        # noqa: E402

import models.mlp_backbone  # noqa: F401,E402  -- triggers registration

PROFILE_S = (0.0, 0.5, 1.0, 1.5, 2.0)
SWEEP = [
    ("checkpoints/stage1_linear_k16_s80000_related0.1.pt", "related, perturb 0.1"),
    ("checkpoints/stage1_linear_k16_s80000_related0.25.pt", "related, perturb 0.25"),
    ("checkpoints/stage1_linear_k16_s80000_related0.5.pt", "related, perturb 0.5"),
    ("checkpoints/stage1_linear_k16_s80000_related1.0.pt", "related, perturb 1.0"),
    ("checkpoints/stage1_linear_k16_s80000_unrelated.pt", "unrelated"),
]


def ci95(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def load(ckpt: str, dev: torch.device):
    """Rebuild from the checkpoint's own fields; k and the decoder change the shapes."""
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = BaseConfig()
    cfg.model.k = sd["k"]
    cfg.model.n_relations = len(RELATIONS)
    model, enc, tr = build(cfg, dev, sd["decoder"])
    model.load_state_dict(sd["model"])
    enc.load_state_dict(sd["encoder"])
    tr.load_state_dict(sd["transport"])
    for m in (model, enc, tr):
        m.eval()

    # the same meta-test tasks the run was evaluated on: same builder, same seeds
    if sd["family"] == "related":
        fam = build_related_task_family(n_train=192, n_test=32, n_components=4,
                                        perturb=sd["perturb"], seed=sd["seed"], device=dev)
    else:
        fam = build_task_family(n_train=192, n_test=32, n_components=4,
                                seed=sd["seed"], device=dev)
    loader = GMMEpisodeLoader(fam["test"], dev, m_source=256, query_batch=1024,
                              k_shots=cfg.episodes.k_shots, seed=sd["seed"] + 1)
    return model, enc, tr, cfg, loader, sd


@torch.no_grad()
def loss_at(model, z, x0, cfg, seed: int, n_noise: int) -> float:
    """Denoising loss for one coordinate, averaged over n_noise draws of (t, eps).

    The seed is the caller's, so every coordinate on this task sees the same draws.
    """
    dev = x0.device
    total = 0.0
    for j in range(n_noise):
        g = torch.Generator(device=dev).manual_seed(seed + j)
        nb = q_sample(model.schedule, x0, generator=g)
        pred = model.eps_hat(nb.x_t, nb.t, z.unsqueeze(0).expand(x0.shape[0], -1))
        total += float(denoising_loss(nb.eps, pred, nb.t, model.schedule,
                                      cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma))
    return total / n_noise


@torch.no_grad()
def panel(ckpt: str, dev: torch.device, n_tasks: int, k_shot: int, n_noise: int,
          profile: bool = True, out=print) -> dict:
    model, enc, tr, cfg, loader, sd = load(ckpt, dev)
    tag = sd["family"] + (f"@{sd['perturb']}" if sd["family"] == "related" else "")
    out(f"\ncheckpoint {os.path.basename(ckpt)}")
    out(f"  k={sd['k']}  decoder={sd['decoder']}  steps={sd['steps']}  family={tag}")

    tasks = []
    for ti in range(n_tasks):
        b = loader.sample(k_shot=k_shot, task_idx=ti)
        rel = None if tr.relation_emb is None else b.relation.reshape(1)
        z_tld = tr(enc(b.src_support).unsqueeze(0), rel).squeeze(0)
        tasks.append({"batch": b, "z": z_tld})

    z_bar = torch.stack([t["z"] for t in tasks]).mean(dim=0)

    # ---- the four-way panel, every coordinate on identical noise ---------------------
    per: dict[str, list[float]] = {kk: [] for kk in ("own", "mean", "shuffled", "zero")}
    for i, t in enumerate(tasks):
        seed = 5000 + 100 * i                       # shared by the four coordinates
        j = (i + 1) % len(tasks)                    # another task's coordinate
        coords = {"own": t["z"], "mean": z_bar,
                  "shuffled": tasks[j]["z"], "zero": torch.zeros_like(t["z"])}
        for name, z in coords.items():
            per[name].append(loss_at(model, z, t["batch"].tgt_query, cfg, seed, n_noise))

    out(f"\n  denoising loss per coordinate, averaged over {len(tasks)} TASKS "
        f"(not over the batch)")
    out(f"    {'coordinate':<34}{'loss':>10}{'95% CI':>12}")
    for name, label in (("own", "this task's own (transport)"),
                        ("mean", "one coordinate shared by all"),
                        ("shuffled", "another task's"),
                        ("zero", "z = 0")):
        mu, h = ci95(per[name])
        out(f"    {label:<34}{mu:>10.4f}{h:>12.4f}")

    def paired(a_key: str, b_key: str) -> tuple[float, float]:
        return ci95([x - y for x, y in zip(per[a_key], per[b_key])])

    out(f"\n  paired differences over tasks")
    out(f"    {'quantity':<34}{'value':>10}{'95% CI':>12}")
    res = {}
    for label, key in (("delta_task = L(mean) - L(own)", ("mean", "own")),
                       ("gain_vs_zero = L(0) - L(own)", ("zero", "own")),
                       ("gain_vs_shuffled", ("shuffled", "own"))):
        mu, h = paired(*key)
        res[label.split(" ")[0]] = (mu, h)
        star = "*" if abs(mu) > h else " "
        out(f"    {label:<34}{mu:>+10.4f}{h:>12.4f} {star}")
    frac = res["delta_task"][0] / res["gain_vs_zero"][0] if res["gain_vs_zero"][0] else float("nan")
    out(f"    {'task-specific share of the gain':<34}{frac:>10.3f}")
    out("      (reported for comparison with the CIFAR panel; the absolute paired "
        "difference above is the quantity to trust, since this ratio's denominator "
        "decays towards zero)")

    # ---- the coordinate-loss profile -------------------------------------------------
    if profile:
        out(f"\n  coordinate-loss profile: L(z_bar + s (z - z_bar)), averaged over tasks")
        out("    s = 0 is the shared mean coordinate, s = 1 is the task's own")
        out(f"\n    {'s':>5}{'toward own':>14}{'toward another':>16}")
        prof_own = {s: [] for s in PROFILE_S}
        prof_other = {s: [] for s in PROFILE_S}
        for i, t in enumerate(tasks):
            seed = 5000 + 100 * i
            j = (i + 1) % len(tasks)
            d_own = t["z"] - z_bar
            d_other = tasks[j]["z"] - z_bar
            for s in PROFILE_S:
                prof_own[s].append(
                    loss_at(model, z_bar + s * d_own, t["batch"].tgt_query, cfg, seed, n_noise))
                prof_other[s].append(
                    loss_at(model, z_bar + s * d_other, t["batch"].tgt_query, cfg, seed, n_noise))
        for s in PROFILE_S:
            mo = sum(prof_own[s]) / len(prof_own[s])
            mt = sum(prof_other[s]) / len(prof_other[s])
            mark = "  <- minimum expected here" if s == 1.0 else ""
            out(f"    {s:>5.1f}{mo:>14.4f}{mt:>16.4f}{mark}")
        best = min(PROFILE_S, key=lambda s: sum(prof_own[s]) / len(prof_own[s]))
        out(f"\n    the profile toward this task's own coordinate is minimised at s = {best:.1f}")
        res["profile_min_s"] = best
        res["profile_own"] = {s: sum(prof_own[s]) / len(prof_own[s]) for s in PROFILE_S}

    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="checkpoints/stage1_linear_k16_s80000_unrelated.pt")
    ap.add_argument("--sweep", action="store_true",
                    help="run the panel across the relatedness family as well")
    ap.add_argument("--n-tasks", type=int, default=32)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-noise", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: this is the control that must not queue behind "
                         "anything on the GPU")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    dev = torch.device(a.device)
    emit("STAGE B -- positive control on two-dimensional Gaussian mixtures")
    emit("=" * 78)
    emit(f"tasks {a.n_tasks}   K_T {a.k_shot}   noise draws per reading {a.n_noise}   device {dev}")
    emit("every coordinate on a task is evaluated on identical (t, eps); differences are")
    emit("paired per task and averaged across tasks, never across the query batch")

    targets = SWEEP if a.sweep else [(a.ckpt, "")]
    results = {}
    for path, label in targets:
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        if not os.path.exists(full):
            emit(f"\n  (missing, skipped: {path})")
            continue
        results[label or os.path.basename(path)] = panel(
            full, dev, a.n_tasks, a.k_shot, a.n_noise, profile=True, out=emit)

    if a.sweep and len(results) > 1:
        emit("\n\nrelatedness sweep: how task distance moves the task-specific term")
        emit(f"  {'family':<26}{'delta_task':>12}{'gain_vs_zero':>14}{'profile min at s':>18}")
        for label, r in results.items():
            emit(f"  {label:<26}{r['delta_task'][0]:>+12.4f}{r['gain_vs_zero'][0]:>+14.4f}"
                 f"{r.get('profile_min_s', float('nan')):>18.1f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
