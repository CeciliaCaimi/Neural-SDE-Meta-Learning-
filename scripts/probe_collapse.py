"""Locate the layer at which the CIFAR encoder coordinate collapse happens.

Over a batch of held-out episodes, look separately at:
  1. pooled per-image h_psi features: are source and target separable, and is there spread
  2. the coordinate z after rho_psi: the same two questions
If (1) has spread and (2) does not, the collapse is in rho_psi (the read-out learned a near
constant) and changing the pooling or read-out suffices; if (1) has none, pooling itself misses it.
"""
from __future__ import annotations
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig
from domains.cifar100 import load_cifar100
from episodes.splits import load_split
from episodes.dataset import EpisodeLoader
from training.loop import build

def rel_spread(vecs: torch.Tensor) -> float:
    """Relative spread across samples: mean pairwise distance divided by mean norm."""
    d = torch.cdist(vecs, vecs)
    off = d[~torch.eye(len(vecs), dtype=torch.bool, device=vecs.device)]
    return float(off.mean() / vecs.norm(dim=1).mean().clamp_min(1e-8))

def main(ckpt: str):
    dev = torch.device("cuda")
    cfg = BaseConfig()
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    # Build from the checkpoint's own config; defaults may have moved since training.
    ck_model = sd["config"]["model"]
    cfg.model.k = ck_model["k"]
    cfg.model.backbone = ck_model["backbone"]
    cfg.model.backbone_kwargs = dict(ck_model["backbone_kwargs"])
    cfg.model.encoder_pooling = ck_model.get("encoder_pooling", "mean")
    cfg.model.n_relations = ck_model["n_relations"]
    model, enc, tr = build(cfg, dev)
    enc.load_state_dict(sd["encoder"]); enc.eval()
    print(f"checkpoint: {os.path.basename(ckpt)}  (step {sd.get('step','?')})")

    raw = load_cifar100(); sp = load_split(os.path.join(_ROOT, cfg.episodes.split_path))
    ld = EpisodeLoader(raw, sp, "test", device=dev,
                       enc_source_images=cfg.episodes.enc_source_images,
                       query_batch=cfg.episodes.query_batch, seed=7)

    src_pool, tgt_pool = [], []          # pooled h features, per episode
    zs_pool, ze_pool = [], []            # z_S and z^enc_T, per episode
    pair_feat, pair_z = [], []           # per episode: relative source-to-target distance
    with torch.no_grad():
        for _ in range(16):
            b = ld.sample(k_shot=20)
            hs = enc.per_element(b.src_support).mean(0)   # pooled h, (feature_dim,)
            ht = enc.per_element(b.tgt_support).mean(0)
            zs = enc.pool_head(hs); ze = enc.pool_head(ht)
            src_pool.append(hs); tgt_pool.append(ht)
            zs_pool.append(zs);  ze_pool.append(ze)
            pair_feat.append(float((hs-ht).norm()/hs.norm().clamp_min(1e-8)))
            pair_z.append(float((zs-ze).norm()/zs.norm().clamp_min(1e-8)))

    src_pool=torch.stack(src_pool); tgt_pool=torch.stack(tgt_pool)
    zs_pool=torch.stack(zs_pool); ze_pool=torch.stack(ze_pool)

    print("\n[layer 1: pooled h_psi features]")
    print(f"  relative spread across episodes: source {rel_spread(src_pool):.3f}  target {rel_spread(tgt_pool):.3f}")
    print(f"  mean source-to-target distance within an episode: {sum(pair_feat)/len(pair_feat):.3f}")
    print(f"  mean norm of h: {src_pool.norm(dim=1).mean():.3f}")

    print("\n[layer 2: the coordinate z after rho_psi]")
    print(f"  relative spread across episodes: z_S {rel_spread(zs_pool):.3f}  z^enc_T {rel_spread(ze_pool):.3f}")
    print(f"  mean z_S-to-z^enc_T distance within an episode: {sum(pair_z)/len(pair_z):.3f}")
    print(f"  mean norm of z: {zs_pool.norm(dim=1).mean():.3f}")

    print("\n[reading]")
    feat_ok = rel_spread(src_pool) > 0.05
    z_ok = rel_spread(zs_pool) > 0.05
    if feat_ok and not z_ok:
        print("  -> h_psi has spread but z does not: the collapse is in the rho_psi read-out")
    elif not feat_ok:
        print("  -> h_psi has no spread: mean pooling misses the difference; use higher moments")
    else:
        print("  -> both layers have spread: the collapse metric needs another explanation")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1
         else os.path.join(_ROOT,"checkpoints","cifar_k16_50k_step5000.pt"))
