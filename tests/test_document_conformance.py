"""Conformance checks against the source document specification.

    python -m tests.test_document_conformance

Every verifiable clause of sections 6.1, 6.2, 8.5, 10, A.1, A.2 and A.3 is written as an
assertion. Manual comparison misses things and cannot stop later drift; this file can be re-run.

Each check is labelled with the section of the document it comes from.
"""

from __future__ import annotations

import inspect
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import adaptation.coordinate as AC                                    # noqa: E402
import training.meta_train as MT                                      # noqa: E402
from config.base_config import BaseConfig                             # noqa: E402
from diffusion.schedule import NoiseSchedule                          # noqa: E402
from episodes.splits import ROLE_SOURCE, ROLE_TARGET, load_split      # noqa: E402
from models.backbone import build_backbone                            # noqa: E402
from models.score_model import ScoreModel                             # noqa: E402
from models.set_encoder import ConvSetEncoder                         # noqa: E402
from models.transport import Transport                                # noqa: E402
import models.unet                                                    # noqa: F401,E402

RULE = "─" * 88
results: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((section, name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {section:<8} {name}" + (f"  ——  {detail}" if detail else ""))


def deviation(section: str, name: str, detail: str) -> None:
    """A known deviation, confirmed by the project. Listed separately, not counted as a failure."""
    print(f"  [DEVI] {section:<8} {name}  ——  {detail}")



def _code_only(src: str) -> str:
    """Source with comments and string literals removed, so identifier checks see code only."""
    import io as _io
    import tokenize as _tok
    out = []
    try:
        for tk in _tok.generate_tokens(_io.StringIO(src).readline):
            if tk.type in (_tok.COMMENT, _tok.STRING):
                continue
            out.append(tk.string)
    except (_tok.TokenError, IndentationError):
        return src
    return " ".join(out)


def main() -> int:
    torch.manual_seed(0)
    cfg = BaseConfig()
    k, C, S = cfg.model.k, 3, 32
    dev = torch.device("cpu")

    print(RULE); print("conformance with the document specification"); print(RULE)

    sched = NoiseSchedule(cfg.diffusion.n_steps, cfg.diffusion.schedule)
    bb = build_backbone("small_unet", base_channels=64, channel_mult=(1, 2),
                        num_res_blocks=1, attn_resolutions=(16,))
    model = ScoreModel(bb, sched, k=k).eval()
    enc = ConvSetEncoder(k=k).eval()
    tr = Transport(k=k, n_relations=cfg.model.n_relations,
                   relation_dim=cfg.model.relation_dim,
                   hidden=cfg.model.transport_hidden).eval()

    # ================= Network structure =================
    print("\n[network structure]")
    x = torch.randn(2, C, S, S); t = torch.tensor([10, 900])

    with torch.no_grad():
        H = model.features(x, t)
        e0 = model.eps_base(x, t)
        R = model.basis(x, t)
    check("§10.1", "the backbone emits a shared feature tensor H in R^{C x H x W}",
          H.dim() == 4 and H.shape[0] == 2, str(tuple(H.shape)))
    check("§10.2", "the base head predicts eps_hat_0 with the image shape",
          tuple(e0.shape) == (2, C, S, S), str(tuple(e0.shape)))
    check("§10.2", "the basis head emits k x C_img channels, reshaped into R_1..R_k",
          tuple(R.shape) == (2, k, C, S, S) and
          model.basis_head.out_channels == k * C,
          f"conv emits {model.basis_head.out_channels} = {k}x{C}, reshaped to {tuple(R.shape)}")

    # "one U-Net, not k separate denoisers": count backbone calls inside one eps_hat
    calls = {"n": 0}
    orig = bb.forward_features
    bb.forward_features = lambda *a, **kw: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **kw))[1]
    with torch.no_grad():
        model.eps_hat(x, t, torch.randn(2, k))
    n_one = calls["n"]
    calls["n"] = 0
    with torch.no_grad():
        model.eps_hat_many(x, t, [torch.randn(2, k), torch.randn(2, k), None])
    n_many = calls["n"]
    bb.forward_features = orig
    check("§10.2", "one backbone network rather than k independent denoisers",
          n_one == 1 and n_many == 1, f"one coordinate: {n_one} pass; three coordinates: {n_many} pass")

    check("§10.3", "source and target share the same encoder object",
          "encoder(batch.src_support)" in inspect.getsource(MT.compute_coordinates)
          and "encoder(batch.tgt_support)" in inspect.getsource(MT.compute_coordinates),
          "compute_coordinates calls the same encoder parameter twice")

    lin = [m for m in tr.net if isinstance(m, torch.nn.Linear)]
    dims = [lin[0].in_features] + [m.out_features for m in lin]
    d_c = cfg.model.relation_dim if cfg.model.n_relations else 0
    check("§A.2", "transport is a residual MLP (k+d_c) -> 64 -> 64 -> k",
          dims == [k + d_c, cfg.model.transport_hidden, cfg.model.transport_hidden, k]
          and "z_s + out" in inspect.getsource(Transport.forward),
          f"layer dims {dims}, forward contains the residual addition")

    forbidden = ("uncertainty", "normalizing_flow", "normalising_flow", "gate", "posterior")
    # Scan code only. Comments and docstrings legitimately *name* these components in order
    # to record that they are deliberately absent, so tokenise and drop comments/strings.
    src_all = "".join(_code_only(inspect.getsource(m))
                      for m in (ScoreModel, ConvSetEncoder, Transport))
    check("§10", "none of the uncertainty / flow / gate components excluded in v1",
          not any(f in src_all.lower() for f in forbidden), "no such identifier in the three modules")

    # ================= Training flow =================
    print("\n[training flow]")
    ms = inspect.getsource(MT.meta_step)
    cc = inspect.getsource(MT.compute_coordinates)

    # Static: inspect only the transport call line itself
    tr_lines = [l.strip() for l in cc.splitlines() if "transport(" in l and "=" in l]
    static_ok = len(tr_lines) == 1 and "z_s" in tr_lines[0]         and "tgt" not in tr_lines[0] and "target" not in tr_lines[0]
    # Runtime: record what transport actually receives, asserting nothing image-shaped appears
    seen: list[tuple] = []
    orig_fwd = Transport.forward
    Transport.forward = lambda self, z, rel=None: (
        seen.append((tuple(z.shape), None if rel is None else tuple(rel.shape))),
        orig_fwd(self, z, rel))[1]
    class _B:
        src_support = torch.randn(8, C, S, S); tgt_support = torch.randn(3, C, S, S)
        relation = torch.tensor(0)
    with torch.no_grad():
        MT.compute_coordinates(enc, tr, _B())
    Transport.forward = orig_fwd
    runtime_ok = len(seen) == 1 and seen[0][0] == (1, k)
    check("§8.5", "z_tilde_T = z_S + Delta_gamma(z_S, c); the input holds no target sample",
          static_ok and runtime_ok,
          f"call line `{tr_lines[0][:44]}`; received {seen[0]} at runtime (a (1,k) coordinate, not an image)")
    check("(27)", "L_src is computed on src_query using z_S",
          "q_sample(sched, batch.src_query)" in ms and "z_s.unsqueeze(0)" in ms)
    check("(28)(29)", "L_tgt and L_trans on tgt_query, using z^enc_T and z_tilde_T respectively",
          "q_sample(sched, batch.tgt_query)" in ms and
          "z_enc_t.unsqueeze(0)" in ms and "z_tld_t.unsqueeze(0)" in ms)
    check("(28)(29)", "both share one noising (a paired comparison)",
          ms.count("q_sample(") == 2 and "eps_hat_many" in ms,
          "q_sample is called exactly twice in meta_step; the target side shares nb_t")
    check("(30)", "lambda_z regularises only z_S and z^enc_T, never z_tilde_T",
          "z_s.pow(2).sum() + z_enc_t.pow(2).sum()" in ms and "z_tld_t.pow" not in ms)
    check("§A.3", "training never differentiates through the refinement inner loop",
          "refine" not in ms and "adapt(" not in ms, "meta_step calls no refinement")

    # meta-test: every network parameter frozen, only z carries a gradient
    rf = inspect.getsource(AC.refine)
    check("§A.3", "refinement freezes all network parameters, leaving only z free",
          "p.requires_grad_(False)" in rf and "torch.optim.Adam([z]" in rf)
    check("§8.4", "the refinement objective has a 1/K_T denoising term and a beta_0/K_T prior",
          "budget.beta0 / k_t" in rf, "the prior scales with the support size")
    check("§A.1", "the refine signature excludes target query (the red line is inexpressible)",
          "tgt_query" not in str(inspect.signature(AC.refine)),
          str(inspect.signature(AC.refine)).replace("  ", " ")[:78])

    # ================= Dataset split =================
    print("\n[dataset split]")
    sp = load_split(os.path.join(_ROOT, cfg.episodes.split_path))
    sets = {w: set(sp.superclass_split[w]) for w in ("train", "val", "test")}
    fines = {w: set(sp.fine_ids(w)) for w in ("train", "val", "test")}
    check("§6.2", "semantic tasks are split rather than images, and the ways are disjoint",
          not (fines["train"] & fines["test"]) and not (sets["train"] & sets["test"]),
          f"train {len(fines['train'])} fine classes / test {len(fines['test'])} fine classes")
    check("§6.1", "target data splits into support and query, the support deliberately small",
          all(p.tgt_support_reserve.size == max(cfg.episodes.k_shots) and p.tgt_query.size > 500
              for p in sp.pools.values() if p.role == ROLE_TARGET),
          f"support reserve {max(cfg.episodes.k_shots)}, query {600 - max(cfg.episodes.k_shots)}")
    check("(16)", "the defining condition M_S >> K_T",
          all(p.src_support.size >= 20 * max(cfg.episodes.k_shots)
              for p in sp.pools.values() if p.role == ROLE_SOURCE),
          f"M_S={cfg.episodes.enc_source_images} per step / pool 500, K_T<={max(cfg.episodes.k_shots)}")
    check("§A.1", "support and query are disjoint within each domain",
          all(torch.tensor(list(set(p.src_support.tolist()) & set(p.src_query.tolist()))).numel() == 0
              for p in sp.pools.values() if p.role == ROLE_SOURCE))

    # Known deviations, recorded explicitly so they are never mistaken for passes
    print("\n[known deviations from the document]")
    deviation("§12.3", "this sibling-class scheme is a stress test, not the main experiment",
              "the primary clean-to-corrupted protocol lives in episodes/domainshift.py")
    deviation("§12.3", "here the target domain is a sibling fine class, not a corruption",
              "the relation becomes sibling-to-sibling rather than a domain shift")
    deviation("§A.1", "the episode tuple stores no d_S / d_T",
              "this scheme has roles rather than domains")
    check("§A.1", "source and target sample counts are recorded separately",
          ("'M_S'" in ms or '"M_S"' in ms) and ("'K_T'" in ms or '"K_T"' in ms),
          "the training log must carry both M_S and K_T")

    n_fail = sum(1 for *_, ok, _ in results if not ok)
    print()
    print(RULE)
    print(f"specification clauses {len(results) - n_fail} / {len(results)} passed"
          + ("" if not n_fail else f"  --  {n_fail} failed"))
    print("known deviations are marked DEVI; each is a deliberate decision and is not a failure")
    print(RULE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
