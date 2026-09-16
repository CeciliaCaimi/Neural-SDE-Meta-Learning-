"""Checks on the source-dependence controls and the nested source sets.

Runs on CPU so it does not contend with a generation job on the GPU. Every check either
raises or prints PASS; a check that can only print is not a check.

    python scripts/check_source_controls.py checkpoints/cifar_ds3_step50000.pt \
        --domainshift-path artifacts/cifar100_domainshift_c3.json
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from adaptation.budget import AdaptBudget                                 # noqa: E402
from adaptation.coordinate import SOURCE_DEPENDENT, adapt                 # noqa: E402
from domains.cifar100 import load_cifar100                                # noqa: E402
from episodes.domainshift import DomainShiftLoader, load_domainshift      # noqa: E402
from posthoc_controls import load_checkpoint                              # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--domainshift-path", required=True)
    a = ap.parse_args()

    dev = torch.device("cpu")
    model, enc, tr, cfg, step, _ = load_checkpoint(a.ckpt, dev)
    ds_path = a.domainshift_path
    if not os.path.isabs(ds_path):
        ds_path = os.path.join(_ROOT, ds_path)
    split = load_domainshift(ds_path)
    cors = tuple(split.config.corruptions)
    raw = load_cifar100()
    budget = AdaptBudget(steps=cfg.adapt.steps, lr=cfg.adapt.lr, beta0=cfg.adapt.beta0,
                         noise_batch=cfg.adapt.noise_batch)
    print(f"checkpoint {os.path.basename(a.ckpt)} step {step}, {len(cors)} transformations: {cors}\n")

    # ---------------------------------------------------------------- 1. nesting ----
    print("1. nested source sets vary size alone")
    sets = {}
    for n in (16, 32, 64):
        ld = DomainShiftLoader(raw, split, "val", device=dev, enc_source_images=n,
                               query_batch=16, seed=7, nested_source=True)
        ld.fids = [ld.fids[0]]
        sets[n] = ld.sample(k_shot=1, corruption=cors[0]).src_support

    check("M_S=16 is the prefix of M_S=32",
          torch.equal(sets[16], sets[32][:16]))
    check("M_S=32 is the prefix of M_S=64",
          torch.equal(sets[32], sets[64][:32]))
    check("sizes are what was asked for",
          [sets[n].shape[0] for n in (16, 32, 64)] == [16, 32, 64],
          f"{[sets[n].shape[0] for n in (16, 32, 64)]}")

    # nesting must NOT hold when the flag is off, or the flag would be doing nothing
    off = {}
    for n in (16, 32):
        ld = DomainShiftLoader(raw, split, "val", device=dev, enc_source_images=n,
                               query_batch=16, seed=7, nested_source=False)
        ld.fids = [ld.fids[0]]
        off[n] = ld.sample(k_shot=1, corruption=cors[0]).src_support
    check("without the flag the draws are independent, as before",
          not torch.equal(off[16], off[32][:16]))

    # ------------------------------------------------------- 2. relation-only -------
    print("\n2. relation-only depends on the relation and on nothing else")
    ld = DomainShiftLoader(raw, split, "val", device=dev,
                           enc_source_images=cfg.episodes.enc_source_images,
                           query_batch=16, seed=11)
    all_fids = ld.fids
    zs_by_cor: dict[str, list[torch.Tensor]] = {c: [] for c in cors}
    for fid in all_fids[:3]:
        ld.fids = [fid]
        for c in cors:
            b = ld.sample(k_shot=1, corruption=c)
            st = adapt("relation_only", model, enc, tr, b, budget, cfg.diffusion.loss_weighting)
            zs_by_cor[c].append(st.z)
    ld.fids = all_fids

    for c in cors:
        zs = zs_by_cor[c]
        same = all(torch.equal(zs[0], z) for z in zs[1:])
        check(f"'{c}': identical across 3 classes", same,
              f"max spread {max(float((z - zs[0]).abs().max()) for z in zs):.2e}")
    if len(cors) > 1:
        distinct = not torch.equal(zs_by_cor[cors[0]][0], zs_by_cor[cors[1]][0])
        check("different transformations give different coordinates", distinct)

    # relation-only must not be the same thing as transport, or it is not a control
    b = ld.sample(k_shot=1, corruption=cors[0])
    z_tr = adapt("transport_no_refine", model, enc, tr, b, budget, cfg.diffusion.loss_weighting).z
    z_ro = adapt("relation_only", model, enc, tr, b, budget, cfg.diffusion.loss_weighting).z
    check("relation-only differs from transport", not torch.equal(z_tr, z_ro),
          f"distance {float((z_tr - z_ro).norm()):.4f}")
    check("relation-only does not refine",
          adapt("relation_only", model, enc, tr, b, budget,
                cfg.diffusion.loss_weighting).steps_taken == 0)

    # ------------------------------------------------------- 3. the override --------
    print("\n3. the source-coordinate override is honoured, and refused where it would be ignored")
    z_fake = torch.randn_like(z_tr)
    z_out = adapt("transport_no_refine", model, enc, tr, b, budget,
                  cfg.diffusion.loss_weighting, z_s_override=z_fake).z
    check("override changes the transported coordinate", not torch.equal(z_out, z_tr))
    check("source_reuse returns the override unchanged",
          torch.equal(adapt("source_reuse", model, enc, tr, b, budget,
                            cfg.diffusion.loss_weighting, z_s_override=z_fake).init_z, z_fake))

    for bad in ("target_only", "zero", "relation_only", "oracle"):
        try:
            adapt(bad, model, enc, tr, b, budget, cfg.diffusion.loss_weighting,
                  z_s_override=z_fake, oracle_data=b.tgt_query if bad == "oracle" else None)
            check(f"'{bad}' refuses an override it would ignore", False)
        except ValueError:
            check(f"'{bad}' refuses an override it would ignore", True)
    check("the refusal list matches the strategies that read z_S",
          SOURCE_DEPENDENT == ("source_reuse", "transport", "transport_no_refine"))

    print()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} check(s) failed: {FAILURES}")
    print("all checks passed")


if __name__ == "__main__":
    main()
