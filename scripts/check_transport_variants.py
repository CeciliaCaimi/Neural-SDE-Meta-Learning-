"""Checks on the A5 transport variants.

Runs on CPU. The round-trip check is the one that matters: every evaluation script
rebuilds its modules from the checkpoint's config and then loads weights into them, so a
variant that does not survive save-and-rebuild is a silent wrong answer waiting to happen.

    python scripts/check_transport_variants.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import asdict

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                                  # noqa: E402
from models.transport import TRANSPORTS, build_transport                   # noqa: E402
from training.loop import build                                            # noqa: E402

FAILURES: list[str] = []
K, N_REL = 16, 3


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    dev = torch.device("cpu")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)   # these are shape and value checks, not training

    print("1. interface and shapes")
    for kind in TRANSPORTS:
        tr = build_transport(kind, k=K, n_relations=N_REL, relation_dim=8, hidden=64,
                             out_moments=False)
        z1 = torch.randn(K)
        zb = torch.randn(5, K)
        rel = torch.tensor(1)
        relb = torch.tensor([0, 1, 2, 1, 0])
        o1, ob = tr(z1.unsqueeze(0), rel.reshape(1)).squeeze(0), tr(zb, relb)
        check(f"{kind}: (k,) in -> (k,) out", tuple(o1.shape) == (K,), str(tuple(o1.shape)))
        check(f"{kind}: (B,k) in -> (B,k) out", tuple(ob.shape) == (5, K), str(tuple(ob.shape)))
        d = tr.delta(zb, relb)
        check(f"{kind}: delta agrees with forward minus z_S",
              torch.allclose(zb + d, ob, atol=1e-6),
              f"max {float((zb + d - ob).abs().max()):.2e}")
        n_par = sum(p.numel() for p in tr.parameters())
        print(f"        parameters: {n_par}")

    print("\n2. every variant starts at 'reuse z_S', so none begins ahead")
    for kind in TRANSPORTS:
        tr = build_transport(kind, k=K, n_relations=N_REL, relation_dim=8, hidden=64,
                             out_moments=False)
        zb = torch.randn(7, K)
        relb = torch.randint(0, N_REL, (7,))
        out = tr(zb, relb)
        moved = float((out - zb).abs().max())
        check(f"{kind}: |z_T - z_S| is ~0 at initialisation", moved < 5e-2, f"max {moved:.2e}")

    print("\n3. the relation is actually consulted (after the parameters move)")
    for kind in ("constant", "linear"):
        tr = build_transport(kind, k=K, n_relations=N_REL, relation_dim=8, hidden=64,
                             out_moments=False)
        with torch.no_grad():                      # nudge the per-relation parameters
            for p in tr.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        z = torch.randn(1, K)
        outs = [tr(z, torch.tensor([r])) for r in range(N_REL)]
        distinct = all(not torch.equal(outs[i], outs[j])
                       for i in range(N_REL) for j in range(i + 1, N_REL))
        check(f"{kind}: different relations give different coordinates", distinct)
    tr = build_transport("identity", k=K, n_relations=N_REL)
    z = torch.randn(1, K)
    check("identity: ignores the relation, by construction",
          torch.equal(tr(z, torch.tensor([0])), tr(z, torch.tensor([2]))))
    check("identity: returns z_S exactly", torch.equal(tr(z, torch.tensor([1])), z))

    print("\n4. round trip: save with one kind, rebuild from the saved config, load weights")
    for kind in TRANSPORTS:
        cfg = BaseConfig()
        cfg.model.k = K
        cfg.model.n_relations = N_REL
        cfg.model.transport_kind = kind
        cfg.model.backbone_kwargs = dict(base_channels=8, channel_mult=(1,),
                                         num_res_blocks=1, attn_resolutions=())
        model, enc, tr = build(cfg, dev)
        with torch.no_grad():                      # move the weights off their init
            for p in tr.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        z = torch.randn(4, K)
        rel = torch.randint(0, N_REL, (4,))
        before = tr(z, rel)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            torch.save({"config": asdict(cfg), "transport": tr.state_dict()}, path)
            sd = torch.load(path, map_location=dev, weights_only=False)

            # rebuild the way every evaluation script does: a fresh config, fields copied
            # from the checkpoint, then load
            cfg2 = BaseConfig()
            ck = sd["config"]["model"]
            cfg2.model.k = ck["k"]
            cfg2.model.n_relations = ck["n_relations"]
            cfg2.model.transport_kind = ck.get("transport_kind", "residual_mlp")
            cfg2.model.backbone_kwargs = dict(ck["backbone_kwargs"])
            _, _, tr2 = build(cfg2, dev)
            tr2.load_state_dict(sd["transport"])
            after = tr2(z, rel)

        check(f"{kind}: the config records the kind",
              sd["config"]["model"].get("transport_kind") == kind,
              str(sd["config"]["model"].get("transport_kind")))
        check(f"{kind}: rebuilt model reproduces the outputs exactly",
              torch.equal(before, after),
              f"max {float((before - after).abs().max()):.2e}")

    print("\n5. a checkpoint of one kind refuses to load into another")
    cfg = BaseConfig()
    cfg.model.k, cfg.model.n_relations = K, N_REL
    cfg.model.backbone_kwargs = dict(base_channels=8, channel_mult=(1,),
                                     num_res_blocks=1, attn_resolutions=())
    cfg.model.transport_kind = "linear"
    _, _, lin = build(cfg, dev)
    cfg.model.transport_kind = "residual_mlp"
    _, _, mlp = build(cfg, dev)
    try:
        mlp.load_state_dict(lin.state_dict())
        check("loading a linear checkpoint into the residual map fails loudly", False)
    except RuntimeError:
        check("loading a linear checkpoint into the residual map fails loudly", True)

    print()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    print("all checks passed")


if __name__ == "__main__":
    main()
