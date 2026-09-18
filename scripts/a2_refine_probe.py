"""A2, follow-up: is the basis's oracle advantage really about optimising z?

    python scripts/a2_refine_probe.py --out handoff/results/A2_refine_probe.txt

A2's generation table says the two conditioning mechanisms are indistinguishable at transport
and that the additive basis is better at the **oracle** -- where the coordinate is fitted by
gradient descent on abundant target data -- at every K_T, against a run-to-run floor about
sixteen times smaller. The obvious explanation is that eps_hat is *linear* in z for the basis,

    eps_hat_z = eps_hat_0 + sum_l z_l R_l(x_t, t),

so refining z is a near-linear least-squares problem, while FiLM's dependence on z is
nonlinear and passes through every residual block, which makes the same descent harder. That
is a hypothesis, and this measures it instead of asserting it.

The measurement: from the same starting coordinate z_tld, run the same refinement objective
with the same budget and the same noise draws in both arms, and record how far the denoising
loss actually descends. Reported as a fraction of each arm's own starting loss, because the
two arms are different networks and their absolute losses are not comparable.

Two controls decide whether any difference is about z and not about the arms differing
anyway:

* **descent from z = 0.** A second run starting at the origin. If the basis descends further
  from *any* start, the effect is about the loss surface in z, which is the claim.
* **the curvature of the surface along the descent direction.** The loss is evaluated at
  several points on the segment from z_tld to the refined z. A quadratic, well-conditioned
  surface descends smoothly; a rugged one does not.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from adaptation.budget import AdaptBudget                                 # noqa: E402
from adaptation.coordinate import refine                                  # noqa: E402
from diffusion.forward import q_sample                                    # noqa: E402
from domains.cifar100 import load_cifar100                                # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift      # noqa: E402
from diffusion.losses import denoising_loss                               # noqa: E402
from posthoc_controls import ci95, episode_coordinates, load_checkpoint   # noqa: E402

ARMS = [("basis (Bz)", "checkpoints/a2_basis_step50000.pt"),
        ("FiLM", "checkpoints/a2_film_step50000.pt")]


@torch.no_grad()
def loss_at(model, z, x0, cfg, seed: int, n_draws: int = 8) -> float:
    """The denoising loss at one coordinate, on a fixed set of (t, eps) draws.

    The draws are seeded identically for every coordinate and every arm, so a difference
    between two numbers is a difference in the coordinate and never in the noise.
    """
    tot = 0.0
    for d in range(n_draws):
        g = torch.Generator(device=x0.device).manual_seed(seed + d)
        nb = q_sample(model.schedule, x0, generator=g)
        zz = z.unsqueeze(0).expand(x0.shape[0], -1)
        pred = model.eps_hat(nb.x_t, nb.t, zz)
        tot += denoising_loss(nb.eps, pred, nb.t, model.schedule,
                              cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma).item()
    return tot / n_draws


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domainshift-path", default="artifacts/cifar100_domainshift_c3.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-classes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    ds = a.domainshift_path if os.path.isabs(a.domainshift_path) \
        else os.path.join(_ROOT, a.domainshift_path)
    split = load_domainshift(ds)
    cors = tuple(split.config.corruptions)
    raw = load_cifar100()

    emit("A2 follow-up: how far does the refinement objective actually descend in each arm?")
    emit("=" * 78)
    emit("Same starting coordinate, same target data, same budget, same (t, eps) draws.")
    emit("Reported as a fraction of each arm's own starting loss: the two arms are different")
    emit("networks and their absolute losses are not comparable.")

    out: dict[str, dict[str, list[float]]] = {}
    for arm, ck in ARMS:
        model, enc, tr, cfg, step, _ = load_checkpoint(os.path.join(_ROOT, ck), dev)
        budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0,
                             noise_batch=cfg.adapt.noise_batch)
        loader = DomainShiftLoader(raw, split, a.split, device=dev,
                                   enc_source_images=cfg.episodes.enc_source_images,
                                   query_batch=128, seed=a.seed)
        fids = split.fine_ids(a.split)[:a.n_classes]
        rel_drop, rel_drop0, rough = [], [], []
        for i, fid in enumerate(fids):
            loader.fids = [fid]
            for ci, c in enumerate(cors):
                b = loader.sample(k_shot=20, corruption=c)
                # not hash(c): string hashing is randomised per process, so a seed built
                # from it is not reproducible across runs even though it is stable within one
                seed = a.seed + 100 * i + 7 * ci
                _, z_tld, _ = episode_coordinates(enc, tr, b, b.tgt_query)
                abundant = b.tgt_query

                # descent from the transported coordinate
                g = torch.Generator(device=dev).manual_seed(seed + 1)
                st = refine(model, z_tld, abundant, budget,
                            weighting=cfg.diffusion.loss_weighting, generator=g)
                l0 = loss_at(model, z_tld, abundant, cfg, seed + 500)
                l1 = loss_at(model, st.z, abundant, cfg, seed + 500)
                rel_drop.append((l0 - l1) / l0)

                # the first control: descent from the origin
                z0 = torch.zeros_like(z_tld)
                g0 = torch.Generator(device=dev).manual_seed(seed + 1)
                st0 = refine(model, z0, abundant, budget,
                             weighting=cfg.diffusion.loss_weighting, generator=g0)
                m0 = loss_at(model, z0, abundant, cfg, seed + 500)
                m1 = loss_at(model, st0.z, abundant, cfg, seed + 500)
                rel_drop0.append((m0 - m1) / m0)

                # the second control: how smooth is the segment it descended along?
                # a well-conditioned surface is monotone along it; measure the departure.
                pts = [loss_at(model, z_tld + s * (st.z - z_tld), abundant, cfg, seed + 500,
                               n_draws=4) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
                diffs = np.diff(pts)
                # Normalised by the starting loss, not by the descent. Dividing by the
                # descent looks natural and is ill-conditioned: where an arm barely moves,
                # the denominator is near zero and the ratio explodes. The first run of this
                # probe reported 4540 for the basis on exactly that mistake.
                rough.append(float(np.sum(np.maximum(diffs, 0.0)) / max(1e-9, pts[0])))
            loader.fids = split.fine_ids(a.split)

        out[arm] = {"from_transport": rel_drop, "from_zero": rel_drop0, "rough": rough}
        emit(f"\n  {arm:<12} {os.path.basename(ck)}  step {step}  {len(rel_drop)} episodes")
        for key, what in (("from_transport", "descent from the transported coordinate"),
                          ("from_zero", "descent from z = 0 (control)"),
                          ("rough", "non-monotone rise along the segment, / starting loss")):
            mu, h = ci95(out[arm][key])
            emit(f"    {what:<52} {mu:+.4f} +-{h:.4f}")

    emit("")
    emit("")
    emit("paired difference, basis minus FiLM (positive = the basis descends further)")
    emit("  the episodes are the same draw in both arms, so this is paired")
    for key, what in (("from_transport", "descent from the transported coordinate"),
                      ("from_zero", "descent from z = 0"),
                      ("rough", "non-monotone rise (positive = the basis is rougher)")):
        d = [x - y for x, y in zip(out["basis (Bz)"][key], out["FiLM"][key])]
        mu, h = ci95(d)
        emit(f"  {what:<52} {mu:+.4f} +-{h:.4f}{'*' if abs(mu) > h else ' '}")
    emit("\n  * = the 95% interval over episodes excludes zero")
    emit("")
    emit("reading it")
    emit("  The hypothesis predicts the basis descends further from both starts, and along a")
    emit("  smoother segment. If it descends further only from the transported coordinate,")
    emit("  the effect is about where transport lands and not about the loss surface. If")
    emit("  neither difference is established, the oracle gap in the generation table needs")
    emit("  another explanation and should be reported without this one.")

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
