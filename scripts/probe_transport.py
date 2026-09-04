"""Coordinate-space diagnostic: does transport move z_S toward an abundant-target reference?

This is the protocol's "source-prior distance to an abundant-target reference".

The existing diagnostic computes z^enc_T from K_T target images; at K_T=1 that is itself a
noisy estimate, so using it as the reference conflates "transport predicted wrongly" with
"the reference is noisy". Here the reference is encoded from many target images instead:

  d_src = ||z_S - z_T_abund|| / ||z_T_abund||        the gap before transport
  d_tld = ||z_tilde_T - z_T_abund|| / ||z_T_abund||  the gap after transport
  closed = d_src - d_tld                             > 0 means transport moves the right way

This is **separate** from whether the basis can express that coordinate: transport may learn
the relation while the basis cannot convert it into a gain. Separating them locates the bottleneck.
"""
from __future__ import annotations
import argparse, math, os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                              # noqa: E402
from domains.cifar100 import load_cifar100                             # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift   # noqa: E402
from training.loop import build                                        # noqa: E402
from training.meta_train import compute_coordinates                    # noqa: E402


@torch.no_grad()
def apply_ema(modules, shadow):
    for m, sh in zip(modules, shadow):
        sd = m.state_dict()
        for k, v in sh.items():
            sd[k].copy_(v.to(sd[k].dtype))


def ci(xs):
    t = torch.tensor(xs)
    return float(t.mean()), 1.96 * float(t.std()) / math.sqrt(len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--m-abundant", type=int, default=230, help="target images used for the reference")
    a = ap.parse_args()

    dev = torch.device("cuda")
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = BaseConfig()
    cfg.model.k = sd["config"]["model"]["k"]
    cfg.model.backbone_kwargs = dict(sd["config"]["model"]["backbone_kwargs"])
    cfg.model.n_relations = sd["config"]["model"]["n_relations"]
    split = load_domainshift(os.path.join(_ROOT, cfg.episodes.domainshift_path))

    model, enc, tr = build(cfg, dev)
    model.load_state_dict(sd["model"]); enc.load_state_dict(sd["encoder"])
    tr.load_state_dict(sd["transport"])
    if sd.get("ema"):
        apply_ema([model, enc, tr], sd["ema"])
    for m in (model, enc, tr):
        m.eval()
    print(f"{os.path.basename(a.ckpt)}  step {sd.get('step','?')}  k={cfg.model.k}  "
          f"split={a.split}")

    raw = load_cifar100()
    # d_src and d_tld do not depend on K_T by construction: z_S and z_tilde_T are built
    # from source data alone. d_enc does depend on it, showing how noisy the sparse target
    # estimate is -- which is precisely why an abundant reference is needed.
    print("")
    print(f"{'K_T':>4} {'d_src':>16} {'d_tld':>16} {'closed':>18} {'d_enc':>16}")
    print("-" * 76)
    for k in a.k_shots:
        ld = DomainShiftLoader(raw, split, a.split, device=dev,
                               enc_source_images=cfg.episodes.enc_source_images,
                               query_batch=a.m_abundant, seed=77)
        ds, dt, de = [], [], []
        with torch.no_grad():
            for _ in range(a.n_episodes):
                b = ld.sample(k_shot=k)
                z_s, z_enc_t, z_tld_t = compute_coordinates(enc, tr, b)
                z_ab = enc(b.tgt_query)                 # the abundant-target reference
                n = z_ab.norm().clamp_min(1e-8)
                ds.append(float((z_s - z_ab).norm() / n))
                dt.append(float((z_tld_t - z_ab).norm() / n))
                de.append(float((z_enc_t - z_ab).norm() / n))
        (ms, hs), (mt, ht) = ci(ds), ci(dt)
        gap = torch.tensor(ds) - torch.tensor(dt)
        mg, hg = ci(gap.tolist())
        flag = "✓" if mg - hg > 0 else ("✗" if mg + hg < 0 else "~")
        me, he = ci(de)
        print(f"{k:>4} {ms:>10.3f}±{hs:.3f} {mt:>10.3f}±{ht:.3f} "
              f"{mg:>+11.3f}±{hg:.3f} {flag} {me:>10.3f}±{he:.3f}")
    print(f"\n({a.n_episodes} {a.split} episodes; reference encoded from {a.m_abundant} target images)")
    print("closed > 0 with a CI excluding zero = transport does move toward the target in z-space")
    print("d_enc = distance from the K_T-sample target encoding to the same reference")


if __name__ == "__main__":
    main()
