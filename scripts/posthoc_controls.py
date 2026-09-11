"""The four-way control panel and the coordinate-loss profile, read off a checkpoint.

    python scripts/posthoc_controls.py checkpoints/cifar_ds3_step50000.pt \
        --domainshift-path artifacts/cifar100_domainshift_c3.json --split val

Why this exists. The mean-coordinate substitution was added to diagnostics/controls.py at
commit ed98afb, *after* every CIFAR run had finished, so no training log on disk contains
it. The two full-width checkpoints have therefore never been measured with the control
that decides the question. This script applies it to a checkpoint that already exists.

It deliberately does **not** import diagnostics/controls.py. That file is being upgraded
independently by another package, and this quantity has been misread twice on this
project; two implementations that can be compared are worth more than one that cannot.

Two departures from the training-time diagnostic, both stated in the output:

  - the coordinate is encoded from one half of the target query pool and the loss is
    measured on the other half, so no coordinate is ever scored on the images that
    produced it;
  - per-episode paired differences are kept rather than only their mean, so an interval
    can be put on the effect. The effect is roughly sixteen times smaller than the
    between-episode spread, which is what makes pairing indispensable here.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                                  # noqa: E402
from diffusion.forward import q_sample                                     # noqa: E402
from diffusion.losses import denoising_loss                                # noqa: E402
from domains.cifar100 import load_cifar100                                 # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift       # noqa: E402
from training.loop import build                                            # noqa: E402

PROFILE_S = (0.0, 0.5, 1.0, 1.5, 2.0)


@torch.no_grad()
def apply_ema(modules, shadow) -> None:
    """EMA shadow weights are a list of dicts in module order (EMA.copy_to, training/loop.py)."""
    for m, sh in zip(modules, shadow):
        sd = m.state_dict()
        for k, v in sh.items():
            sd[k].copy_(v.to(sd[k].dtype))


def ci95(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% interval over episodes."""
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, 1.96 * math.sqrt(var / n)


def load_checkpoint(path: str, dev: torch.device):
    """Rebuild every module from the checkpoint's **own** config, never from current
    defaults: k, backbone width, pooling and relation count all change the shapes, and a
    config that has drifted since training would either fail obscurely or, worse, load a
    mismatched model in silence."""
    sd = torch.load(path, map_location=dev, weights_only=False)
    cfg = BaseConfig()
    cfg.episodes.scheme = "domainshift"
    ck = sd["config"]["model"]
    cfg.model.k = ck["k"]
    cfg.model.backbone = ck["backbone"]
    cfg.model.backbone_kwargs = dict(ck["backbone_kwargs"])
    cfg.model.encoder_pooling = ck.get("encoder_pooling", "mean")
    cfg.model.n_relations = ck["n_relations"]
    cfg.episodes.enc_source_images = sd["config"]["episodes"]["enc_source_images"]

    model, enc, tr = build(cfg, dev)
    model.load_state_dict(sd["model"])
    enc.load_state_dict(sd["encoder"])
    tr.load_state_dict(sd["transport"])
    used_ema = sd.get("ema") is not None
    if used_ema:
        apply_ema([model, enc, tr], sd["ema"])
    for m in (model, enc, tr):
        m.eval()
    return model, enc, tr, cfg, sd.get("step", "?"), used_ema


@torch.no_grad()
def episode_coordinates(enc, tr, batch, half_a):
    """The coordinates this episode can offer.

    z_tld : transport's prediction from source data alone -- the deployment coordinate,
            and the one the training-time diagnostic calls "correct"
    z_abund: the encoder on an abundant target sample -- the strongest coordinate the
            encoder can produce, and an upper bound on what any of this could deliver
    """
    z_s = enc(batch.src_support)
    rel = None if tr.relation_emb is None else batch.relation.reshape(1)
    z_tld = tr(z_s.unsqueeze(0), rel).squeeze(0)
    z_abund = enc(half_a)
    return z_s, z_tld, z_abund


@torch.no_grad()
def losses_on(model, x0, zs: dict[str, torch.Tensor], n_noise: int, seed: int, cfg):
    """Every coordinate in `zs` scored on one shared set of (t, eps) draws.

    Sharing the noising is the whole point: the between-episode spread of the raw loss is
    about sixteen times the effect, so an unpaired comparison over a handful of episodes
    returns an arbitrary sign.
    """
    n = x0.shape[0]
    w, gamma = cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma
    gen = torch.Generator(device=x0.device)
    out = {k: 0.0 for k in zs}
    r_basis = 0.0
    for d in range(n_noise):
        gen.manual_seed(seed + d)
        nb = q_sample(model.schedule, x0, generator=gen)
        names = list(zs)
        preds = model.eps_hat_many(nb.x_t, nb.t, [zs[k].unsqueeze(0).expand(n, -1) for k in names])
        for name, pred in zip(names, preds):
            out[name] += float(denoising_loss(nb.eps, pred, nb.t, model.schedule, w, gamma)) / n_noise
        ref = zs.get("own")
        if ref is not None:
            r_basis += float(model.basis_usage(nb.x_t, nb.t, ref.unsqueeze(0).expand(n, -1)).mean()) / n_noise
    return out, r_basis


@torch.no_grad()
def losses_at_t(model, x0, zs: dict, t_val: int, n_noise: int, seed: int, cfg):
    """The same comparison, but with the timestep pinned instead of sampled.

    Training draws t uniformly, so the headline effect is an average over the whole noise
    schedule. Pinning t asks a different question: *where* in the schedule does knowing
    the task pay? The prediction under test is that it pays only where the transformation
    can no longer be read off the noised image itself.
    """
    n = x0.shape[0]
    w, gamma = cfg.diffusion.loss_weighting, cfg.diffusion.min_snr_gamma
    t = torch.full((n,), int(t_val), device=x0.device, dtype=torch.long)
    gen = torch.Generator(device=x0.device)
    out = {k: 0.0 for k in zs}
    for d in range(n_noise):
        gen.manual_seed(seed + d)
        nb = q_sample(model.schedule, x0, t=t, generator=gen)
        names = list(zs)
        preds = model.eps_hat_many(nb.x_t, nb.t,
                                   [zs[k].unsqueeze(0).expand(n, -1) for k in names])
        for name, pred in zip(names, preds):
            out[name] += float(denoising_loss(nb.eps, pred, nb.t, model.schedule, w, gamma)) / n_noise
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--domainshift-path", default=None,
                    help="must be the split the checkpoint was trained on")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"),
                    help="protocol: read on val; test is for the final number only")
    ap.add_argument("--n-episodes", type=int, default=24)
    ap.add_argument("--n-noise", type=int, default=4, help="(t, eps) draws per episode")
    ap.add_argument("--query-batch", type=int, default=64,
                    help="target query images per episode; split in half, one half to "
                         "encode the abundant coordinate and one half to score on")
    ap.add_argument("--k-shot", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--by-timestep", action="store_true",
                    help="where in the noise schedule does the coordinate pay?")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, enc, tr, cfg, step, used_ema = load_checkpoint(a.ckpt, dev)

    ds_path = a.domainshift_path or cfg.episodes.domainshift_path
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(_ROOT, ds_path)
    split = load_domainshift(ds_path)
    n_cor, trained_cor = len(split.config.corruptions), (cfg.model.n_relations or 1)
    if n_cor != trained_cor:
        raise SystemExit(
            f"split/checkpoint mismatch: {os.path.basename(ds_path)} has {n_cor} "
            f"corruption(s) {split.config.corruptions}, the checkpoint was trained with "
            f"{trained_cor}. Pass the split this checkpoint was trained on.")

    print(f"checkpoint {os.path.basename(a.ckpt)}  step {step}  "
          f"weights {'EMA' if used_ema else 'raw'}")
    print(f"split      {os.path.basename(ds_path)}  {split.config.corruptions}  "
          f"severity {split.config.severity}")
    print(f"protocol   {a.n_episodes} {a.split} episodes x {a.n_noise} noise draws, "
          f"K_T={a.k_shot}, M_S={cfg.episodes.enc_source_images}")
    print(f"           coordinate encoded on {a.query_batch // 2} target images, "
          f"loss measured on {a.query_batch - a.query_batch // 2} disjoint ones\n")

    raw = load_cifar100()
    loader = DomainShiftLoader(raw, split, a.split, device=dev,
                               enc_source_images=cfg.episodes.enc_source_images,
                               query_batch=a.query_batch, seed=a.seed)

    # ---- pass 1: episodes and their coordinates -------------------------------------
    eps_data = []
    for i in range(a.n_episodes):
        b = loader.sample(k_shot=a.k_shot)
        h = b.tgt_query.shape[0] // 2
        half_a, half_b = b.tgt_query[:h], b.tgt_query[h:]
        z_s, z_tld, z_abund = episode_coordinates(enc, tr, b, half_a)
        eps_data.append(dict(batch=b, eval_x0=half_b, z_s=z_s, z_tld=z_tld, z_abund=z_abund))

    z_bar = torch.stack([e["z_tld"] for e in eps_data]).mean(dim=0)
    z_bar_abund = torch.stack([e["z_abund"] for e in eps_data]).mean(dim=0)

    # A task here is (semantic class, transformation). Averaging over *all* episodes
    # removes both; averaging only over episodes that share a transformation removes the
    # class and keeps the transformation. Comparing the two separates what the coordinate
    # actually carries, which no previous measurement on this project has done.
    by_cor: dict[str, list[int]] = {}
    for i, e in enumerate(eps_data):
        by_cor.setdefault(e["batch"].provenance["corruption"], []).append(i)
    z_bar_cor = {c: torch.stack([eps_data[i]["z_tld"] for i in idx]).mean(dim=0)
                 for c, idx in by_cor.items()}

    # ---- pass 2: the panel ----------------------------------------------------------
    per_ep: dict[str, list[float]] = {}
    rbs: list[float] = []
    for i, e in enumerate(eps_data):
        j = (i + 1) % len(eps_data)
        zs = {
            "own":        e["z_tld"],                       # transport, the deployment coordinate
            "mean":       z_bar,                            # one coordinate shared by every episode
            "shuffled":   eps_data[j]["z_tld"],             # another episode's
            "zero":       torch.zeros_like(e["z_tld"]),
            "abund":      e["z_abund"],                     # encoder on abundant target
            "abund_mean": z_bar_abund,
            # same transformation, class averaged away
            "mean_cor":   z_bar_cor[e["batch"].provenance["corruption"]],
        }
        vals, rb = losses_on(model, e["eval_x0"], zs, a.n_noise, a.seed + 1000 * i, cfg)
        for k, v in vals.items():
            per_ep.setdefault(k, []).append(v)
        rbs.append(rb)

    def diff(a_key: str, b_key: str) -> list[float]:
        return [x - y for x, y in zip(per_ep[a_key], per_ep[b_key])]

    print("denoising loss per coordinate (mean over episodes)")
    print(f"  {'coordinate':<34}{'loss':>10}")
    print("  " + "-" * 44)
    for k, label in (("own", "this episode's own (transport)"),
                     ("mean", "one shared mean coordinate"),
                     ("shuffled", "another episode's"),
                     ("zero", "z = 0"),
                     ("abund", "encoder on abundant target"),
                     ("abund_mean", "shared mean of those")):
        print(f"  {label:<34}{sum(per_ep[k])/len(per_ep[k]):>10.4f}")

    print("\npaired differences over episodes, positive = the coordinate helps")
    print(f"  {'quantity':<34}{'value':>10}  {'95% CI':>9}   verdict")
    print("  " + "-" * 70)
    rows = [
        ("delta_task  = L(mean) - L(own)", diff("mean", "own")),
        ("gain_vs_zero = L(0) - L(own)", diff("zero", "own")),
        ("gain_vs_shuffled", diff("shuffled", "own")),
        ("delta_task, abundant coordinate", diff("abund_mean", "abund")),
        ("  of which: transformation id", diff("mean", "mean_cor")),
        ("  of which: class id", diff("mean_cor", "own")),
    ]
    for label, d in rows:
        mu, h = ci95(d)
        verdict = "> 0" if mu - h > 0 else ("< 0" if mu + h < 0 else "interval contains 0")
        print(f"  {label:<34}{mu:>+10.5f}  +-{h:>7.5f}   {verdict}")

    rb_mu, rb_h = ci95(rbs)
    print(f"\n  r_basis (||B z|| / ||eps_0||){'':6}{rb_mu:>10.4f}  +-{rb_h:>7.4f}")
    z_stack = torch.stack([e["z_tld"] for e in eps_data])
    off = torch.cdist(z_stack, z_stack)
    off = off[~torch.eye(len(eps_data), dtype=torch.bool, device=z_stack.device)]
    print(f"  z_spread_rel{'':22}{float(off.mean()/z_stack.norm(dim=1).mean()):>10.4f}")

    # ---- pass 3: the coordinate-loss profile ----------------------------------------
    if not a.no_profile:
        print("\ncoordinate-loss profile: L(z_bar + s (z - z_bar)), mean over episodes")
        print("  s = 0 is the shared mean coordinate, s = 1 is the episode's own")
        print(f"\n  {'s':>5}{'toward own':>14}{'toward another':>16}")
        print("  " + "-" * 35)
        prof_own: dict[float, list[float]] = {s: [] for s in PROFILE_S}
        prof_oth: dict[float, list[float]] = {s: [] for s in PROFILE_S}
        for i, e in enumerate(eps_data):
            j = (i + 1) % len(eps_data)
            d_own = e["z_tld"] - z_bar
            d_oth = eps_data[j]["z_tld"] - z_bar
            zs = {}
            for s in PROFILE_S:
                zs[f"own@{s}"] = z_bar + s * d_own
                zs[f"oth@{s}"] = z_bar + s * d_oth
            vals, _ = losses_on(model, e["eval_x0"], zs, a.n_noise, a.seed + 1000 * i, cfg)
            for s in PROFILE_S:
                prof_own[s].append(vals[f"own@{s}"])
                prof_oth[s].append(vals[f"oth@{s}"])
        for s in PROFILE_S:
            mo = sum(prof_own[s]) / len(prof_own[s])
            mt = sum(prof_oth[s]) / len(prof_oth[s])
            print(f"  {s:>5.1f}{mo:>14.4f}{mt:>16.4f}")
        flat = max(abs(sum(prof_own[s]) / len(prof_own[s]) - sum(prof_own[0.0]) / len(prof_own[0.0]))
                   for s in PROFILE_S)
        rise = max(abs(sum(prof_oth[s]) / len(prof_oth[s]) - sum(prof_oth[0.0]) / len(prof_oth[0.0]))
                   for s in PROFILE_S)
        print(f"\n  largest excursion along the correct ray  {flat:.5f}")
        print(f"  largest excursion along a wrong ray      {rise:.5f}")

    # ---- pass 4: where in the noise schedule does the coordinate pay? ---------------
    if a.by_timestep:
        ab = model.schedule.alphas_cumprod
        print("")
        print("where the coordinate pays, by noise level")
        print("  t = diffusion step; sigma = how much of x_t is noise rather than image")
        print("")
        hdr = f"  {'t':>5}{'sigma':>8}{'L(own)':>10}{'delta_task':>13}{'95% CI':>11}{'% of L':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for tv in (25, 100, 250, 400, 550, 700, 850, 975):
            owns, means = [], []
            for i, e in enumerate(eps_data):
                vals = losses_at_t(model, e["eval_x0"], {"own": e["z_tld"], "mean": z_bar},
                                   tv, a.n_noise, a.seed + 1000 * i, cfg)
                owns.append(vals["own"])
                means.append(vals["mean"])
            mu, h = ci95([m - o for m, o in zip(means, owns)])
            lo = sum(owns) / len(owns)
            sigma = float((1.0 - ab[tv]).clamp_min(0).sqrt())
            print(f"  {tv:>5}{sigma:>8.3f}{lo:>10.4f}{mu:>+13.5f}{h:>11.5f}"
                  f"{100 * mu / max(lo, 1e-9):>8.2f}%")


if __name__ == "__main__":
    main()
