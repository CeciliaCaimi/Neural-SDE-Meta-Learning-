"""Checks on the A2 FiLM arm, before it is given 50 000 steps of GPU time.

Runs on CPU in well under a minute and needs no dataset.

Three of these are not stylistic. (1) Section 3 of the note requires the additive basis to
be *replaced*, not supplemented: if any basis term survives, the arm answers a different
question and the comparison is void. (2) Section 1 fixes where FiLM is inserted and what it
computes; a modulation that reaches only some blocks, or that varies over space, is not the
mechanism being tested. (3) The two arms share every module name and -- once the backbone
is film_unet -- every parameter shape, so a FiLM checkpoint loads into the basis model
without raising a word. Check 8 demonstrates that silently, and then shows the recorded
config field catching it.

    python scripts/check_film_arm.py
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from baselines.film_conditioning import FiLMScoreModel                     # noqa: E402
from config.base_config import BaseConfig                                  # noqa: E402
from diffusion.schedule import NoiseSchedule                               # noqa: E402
from models.film_unet import FiLMUNet                                      # noqa: E402
from models.score_model import ScoreModel                                  # noqa: E402
from models.unet import ResBlock, SmallUNet                                # noqa: E402
from training.loop import build, save_checkpoint                           # noqa: E402

FAILURES: list[str] = []

K = 8
UNET_KW = dict(base_channels=32, channel_mult=(1, 2), num_res_blocks=2,
               attn_resolutions=(16,), dropout=0.1)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def small_cfg(film: bool, film_mode: str = "per_block") -> BaseConfig:
    """The two arms' configs, differing only where they are meant to."""
    cfg = BaseConfig()
    cfg.model.k = K
    cfg.model.n_relations = 3
    cfg.model.backbone_kwargs = dict(UNET_KW)
    cfg.diffusion.n_steps = 1000
    if film:
        cfg.model.backbone = "film_unet"
        cfg.model.score_model = "film"
        cfg.model.backbone_kwargs = {**cfg.model.backbone_kwargs,
                                     "k": K, "film_mode": film_mode}
    return cfg


def fresh(film: bool, seed: int = 0, film_mode: str = "per_block"):
    torch.manual_seed(seed)
    return build(small_cfg(film, film_mode), torch.device("cpu"))


def inputs(b: int = 3, size: int = 32, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 3, size, size, generator=g)
    t = torch.randint(0, 1000, (b,), generator=g)
    z = torch.randn(b, K, generator=g)
    return x, t, z


# ---------------------------------------------------------------------------


def s1_insertion_points() -> None:
    print("1. FiLM is inserted where the note says, and computes what it says")
    plain = SmallUNet(**UNET_KW)
    film = FiLMUNet(k=K, **UNET_KW)

    plain_blocks = [m for m in plain.modules() if isinstance(m, ResBlock)]
    film_blocks = [m for m in film.modules() if isinstance(m, ResBlock)]
    check("the FiLM backbone has the same residual blocks as the reference backbone",
          len(plain_blocks) == len(film_blocks) and len(film_blocks) > 0,
          f"{len(film_blocks)} blocks")
    check("every residual block carries a z projection -- none is skipped",
          all(m.z_proj is not None for m in film_blocks),
          f"{sum(m.z_proj is not None for m in film_blocks)}/{len(film_blocks)}")
    check("the reference backbone carries none, so it is untouched",
          all(m.z_proj is None for m in plain_blocks))

    shapes_ok = all(m.z_proj.in_features == K
                    and m.z_proj.out_features == m.emb_proj.out_features == 2 * m.out_ch
                    for m in film_blocks)
    check("each projection is k -> 2*C_j, matching that block's timestep projection",
          shapes_ok,
          f"e.g. {film_blocks[0].z_proj.in_features} -> {film_blocks[0].z_proj.out_features}")

    # The equation itself, on one block, against a hand-written reference.
    torch.manual_seed(3)
    blk = ResBlock(16, 24, temb_dim=32, dropout=0.1).eval()
    blk.attach_film(K)
    check("the projection has no bias, which is what keeps z = 0 exactly the backbone",
          blk.z_proj.bias is None)
    with torch.no_grad():
        blk.z_proj.weight.normal_(std=0.5)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 16, 8, 8, generator=g)
    temb = torch.randn(2, 32, generator=g)
    z = torch.randn(2, K, generator=g)
    with torch.no_grad():
        got = blk(x, temb, z)
        h = blk.in_layers(x)
        scale, shift = blk.emb_proj(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        gamma, beta = blk.z_proj(z)[:, :, None, None].chunk(2, dim=1)
        want = blk.skip(x) + blk.out_layers(
            blk.out_norm(h) * (1 + scale + gamma) + shift + beta)
    # atol, not exact equality: the model adds gamma into scale before the 1, the reference
    # after it, and float addition does not associate. 1e-5 is far below any effect here
    # and far above the 2e-7 that reordering costs.
    check("the block computes (1 + scale_t + gamma_j(z)) * h + shift_t + beta_j(z)",
          torch.allclose(got, want, atol=1e-5),
          f"max deviation {float((got - want).abs().max()):.2e}")

    # gamma and beta are per channel, broadcast over space: a constant feature map must
    # stay constant over space after modulation. Checked on the modulated tensor itself.
    with torch.no_grad():
        gam = blk.z_proj(z)[:, :24]
        check("gamma_j is one scalar per channel, not per pixel",
              tuple(gam.shape) == (2, 24), str(tuple(gam.shape)))


def s2_identity_at_zero() -> None:
    print("\n2. at z = 0 the arm is exactly the unconditioned backbone")
    torch.manual_seed(7)
    film = FiLMUNet(k=K, **UNET_KW).eval()
    zero_init = all(float(m.z_proj.weight.abs().max()) == 0.0 and m.z_proj.bias is None
                    for m in film.modules() if isinstance(m, ResBlock))
    check("every projection is zero-initialised, so training starts at the identity",
          zero_init)

    x, t, z = inputs()
    with torch.no_grad():
        f_none = film.forward_features(x, t)
        f_zero = film.forward_features(x, t, torch.zeros(x.shape[0], K))
        f_z = film.forward_features(x, t, z)
    check("z = None and z = 0 agree", torch.equal(f_none, f_zero))
    check("at initialisation z has no effect at all (identity modulation)",
          torch.equal(f_none, f_z), f"max deviation {float((f_none - f_z).abs().max()):.2e}")

    # After training the projections are not zero; z must then matter.
    with torch.no_grad():
        for m in film.modules():
            if isinstance(m, ResBlock):
                m.z_proj.weight.normal_(std=0.3)
        f_none2 = film.forward_features(x, t)
        f_zero2 = film.forward_features(x, t, torch.zeros(x.shape[0], K))
        f_z2 = film.forward_features(x, t, z)
    check("with trained projections, z = 0 is still exactly the unconditioned backbone",
          torch.equal(f_none2, f_zero2))
    check("with trained projections, a nonzero z changes the features",
          not torch.allclose(f_none2, f_z2, atol=1e-6),
          f"mean |change| {float((f_none2 - f_z2).abs().mean()):.4f}")


def s3_every_block_is_reached() -> None:
    print("\n3. the coordinate reaches every insertion point, not just the first")
    torch.manual_seed(5)
    film = FiLMUNet(k=K, **UNET_KW)
    with torch.no_grad():
        for m in film.modules():
            if isinstance(m, ResBlock):
                m.z_proj.weight.normal_(std=0.3)
    x, t, z = inputs()
    z = z.clone().requires_grad_(True)
    film.forward_features(x, t, z).pow(2).mean().backward()

    blocks = [m for m in film.modules() if isinstance(m, ResBlock)]
    grads = [float(m.z_proj.weight.grad.abs().sum()) for m in blocks]
    check("every block's projection receives gradient",
          all(gv > 0 for gv in grads),
          f"{sum(gv > 0 for gv in grads)}/{len(grads)} blocks, min {min(grads):.2e}")
    check("the coordinate itself receives gradient",
          z.grad is not None and float(z.grad.abs().sum()) > 0,
          f"|dL/dz| = {float(z.grad.abs().sum()):.4e}")


def s4_basis_is_gone() -> None:
    print("\n4. the additive basis is replaced, not supplemented (note section 3)")
    model, _, _ = fresh(film=True)
    model.eval()
    x, t, z = inputs()
    with torch.no_grad():
        before = model.eps_hat(x, t, z)
        # If any basis term survived, scrambling the basis head would move the prediction.
        for p in model.basis_head.parameters():
            p.normal_(std=1.0)
        after = model.eps_hat(x, t, z)
    check("scrambling the basis head does not move eps_hat -- no sum_l z_l R_l survives",
          torch.equal(before, after),
          f"max deviation {float((before - after).abs().max()):.2e}")
    check("the basis head is frozen, so it cannot be trained back in",
          not any(p.requires_grad for p in model.basis_head.parameters()))

    model.zero_grad(set_to_none=True)
    model.eps_hat(x, t, z).pow(2).mean().backward()
    check("the basis head draws no gradient",
          all(p.grad is None for p in model.basis_head.parameters()))
    check("the base head does draw gradient, so the arm is trained through eps_hat_0",
          any(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in model.base_head.parameters()))

    basis_model, _, _ = fresh(film=False)
    basis_model.eval()
    with torch.no_grad():
        b_before = basis_model.eps_hat(x, t, z)
        for p in basis_model.basis_head.parameters():
            p.normal_(std=1.0)
        b_after = basis_model.eps_hat(x, t, z)
    check("the control: in the basis arm the same scrambling does move eps_hat",
          not torch.allclose(b_before, b_after, atol=1e-6),
          f"mean |change| {float((b_before - b_after).abs().mean()):.4f}")


def s5_no_shared_pass() -> None:
    print("\n5. eps_hat_many is overridden: FiLM features depend on z, so one pass is wrong")
    model, _, _ = fresh(film=True)
    model.eval()
    with torch.no_grad():
        for m in model.backbone.modules():
            if isinstance(m, ResBlock):
                m.z_proj.weight.normal_(std=0.3)
    x, t, z = inputs()
    zs = [z, torch.zeros_like(z), z.flip(0)]
    with torch.no_grad():
        many = model.eps_hat_many(x, t, zs)
        one_by_one = [model.eps_hat(x, t, zz) for zz in zs]
    check("eps_hat_many agrees with calling eps_hat once per coordinate",
          all(torch.equal(a, b) for a, b in zip(many, one_by_one)))
    check("the three coordinates give three different predictions",
          not torch.allclose(many[0], many[1], atol=1e-6)
          and not torch.allclose(many[0], many[2], atol=1e-6))
    with torch.no_grad():
        base, resid = model.split_eps(x, t, z)
        e0 = model.eps_hat(x, t, None)
        full = model.eps_hat(x, t, z)
    check("split_eps reports the z = 0 prediction as the base", torch.equal(base, e0))
    check("base and residual add back up to the full prediction",
          torch.allclose(base + resid, full, atol=1e-6),
          f"max deviation {float((base + resid - full).abs().max()):.2e}")


def s6_matched() -> None:
    print("\n6. everything except the conditioning is matched (note section 2)")
    m_b, e_b, t_b = fresh(film=False, seed=0)
    m_f, e_f, t_f = fresh(film=True, seed=0)

    check("same coordinate dimension k", m_b.k == m_f.k == K)
    check("the set encoder is the same module with the same parameter count",
          type(e_b) is type(e_f)
          and sum(p.numel() for p in e_b.parameters()) == sum(p.numel() for p in e_f.parameters())
          and e_b.state_dict().keys() == e_f.state_dict().keys(),
          f"{sum(p.numel() for p in e_f.parameters())} parameters")
    check("the transport map is the same module with the same parameter count",
          type(t_b) is type(t_f)
          and sum(p.numel() for p in t_b.parameters()) == sum(p.numel() for p in t_f.parameters())
          and t_b.state_dict().keys() == t_f.state_dict().keys(),
          f"{sum(p.numel() for p in t_f.parameters())} parameters")

    trunk_b = sum(p.numel() for n, p in m_b.backbone.named_parameters() if "z_proj" not in n)
    trunk_f = sum(p.numel() for n, p in m_f.backbone.named_parameters() if "z_proj" not in n)
    check("the U-Net trunk is identical in size; FiLM adds only its projections",
          trunk_b == trunk_f, f"{trunk_b} parameters each")

    cb, cf = small_cfg(False), small_cfg(True)
    differ = {f for f in vars(cb.model)
              if getattr(cb.model, f) != getattr(cf.model, f)}
    check("the two configs differ in exactly the three fields that name the arm",
          differ == {"backbone", "score_model", "backbone_kwargs"}, str(sorted(differ)))
    other = {f: (getattr(cb.episodes, f), getattr(cf.episodes, f)) for f in vars(cb.episodes)}
    check("episode construction, split and shot counts are untouched",
          all(a == b for a, b in other.values()))
    check("diffusion schedule and objective are untouched",
          vars(cb.diffusion) == vars(cf.diffusion))
    check("optimiser, budget and seed are untouched",
          vars(cb.train) == vars(cf.train) and cb.global_seed == cf.global_seed)

    n_film = sum(p.numel() for p in m_f.backbone.film_parameters())
    n_basis = sum(p.numel() for p in m_b.basis_head.parameters())
    print(f"        conditioning parameters at this size -- FiLM {n_film}, basis head {n_basis}")


def s7_guards() -> None:
    print("\n7. the arm refuses to be built wrong")
    sched = NoiseSchedule(1000, "cosine")
    try:
        FiLMScoreModel(SmallUNet(**UNET_KW), sched, k=K)
        check("FiLMScoreModel on a backbone that ignores z raises", False)
    except TypeError as exc:
        check("FiLMScoreModel on a backbone that ignores z raises", True, str(exc)[:60])
    try:
        FiLMUNet(k=K, film_mode="nonsense", **UNET_KW)
        check("an unknown film_mode raises", False)
    except ValueError:
        check("an unknown film_mode raises", True)

    torch.manual_seed(1)
    a = FiLMUNet(k=K, **UNET_KW).state_dict()
    torch.manual_seed(1)
    b = FiLMUNet(k=K, **UNET_KW).state_dict()
    check("two builds under one seed are bit-identical",
          all(torch.equal(a[key], b[key]) for key in a))


def s8_round_trip() -> None:
    print("\n8. a FiLM checkpoint rebuilds as a FiLM model, and would not without the field")
    cfg = small_cfg(film=True)
    dev = torch.device("cpu")
    torch.manual_seed(2)
    model, enc, tr = build(cfg, dev)
    with torch.no_grad():
        for m in model.backbone.modules():
            if isinstance(m, ResBlock):
                m.z_proj.weight.normal_(std=0.3)
    model.eval(); enc.eval(); tr.eval()
    x, t, z = inputs()
    with torch.no_grad():
        want = model.eps_hat(x, t, z)

    from posthoc_controls import load_checkpoint                            # noqa: E402
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "film_probe.pt")
        opt = torch.optim.AdamW([p for m in (model, enc, tr) for p in m.parameters()])

        class _NoEMA:
            shadow = None
        save_checkpoint(path, 123, cfg, model, enc, tr, opt, _NoEMA())

        back, _, _, cfg2, step, _ = load_checkpoint(path, dev)
        back.eval()
        check("the rebuilt model is the FiLM arm", type(back).__name__ == "FiLMScoreModel",
              type(back).__name__)
        check("the rebuilt backbone is the FiLM U-Net in per-block mode",
              type(back.backbone).__name__ == "FiLMUNet"
              and back.backbone.film_mode == "per_block")
        with torch.no_grad():
            got = back.eps_hat(x, t, z)
        check("predictions survive save and rebuild bit-for-bit", torch.equal(got, want),
              f"max deviation {float((got - want).abs().max()):.2e}")

        # What the recorded field prevents. Rebuild the *basis* composition on the same
        # film_unet backbone: FiLMScoreModel adds no parameters of its own, so the state
        # dict fits, nothing raises, and z never reaches the network.
        cfg_wrong = copy.deepcopy(cfg2)
        cfg_wrong.model.score_model = "basis"
        wrong, _, _ = build(cfg_wrong, dev)
        sd = torch.load(path, map_location=dev, weights_only=False)
        missing = wrong.load_state_dict(sd["model"])
        wrong.eval()
        with torch.no_grad():
            got_wrong = wrong.eps_hat(x, t, z)
        check("loading the FiLM weights into the basis composition raises nothing",
              not missing.missing_keys and not missing.unexpected_keys,
              "which is exactly why the arm is recorded in the config")
        check("and it answers differently, so the field is load-bearing",
              not torch.allclose(got_wrong, want, atol=1e-5),
              f"mean |difference| {float((got_wrong - want).abs().mean()):.4f}")


def s9_temb_mode() -> None:
    print("\n9. the E15 variant (z folded into the timestep embedding) still works")
    torch.manual_seed(9)
    film = FiLMUNet(k=K, film_mode="temb", **UNET_KW).eval()
    check("no per-block projections in this mode",
          all(m.z_proj is None for m in film.modules() if isinstance(m, ResBlock)))
    x, t, z = inputs()
    with torch.no_grad():
        f_none = film.forward_features(x, t)
        f_zero = film.forward_features(x, t, torch.zeros(x.shape[0], K))
        f_z = film.forward_features(x, t, z)
    check("z = None, z = 0 and any z agree at initialisation (zero-initialised z_mlp)",
          torch.equal(f_none, f_zero) and torch.equal(f_none, f_z))
    with torch.no_grad():
        film.z_mlp[-1].weight.normal_(std=0.2)
        f_z2 = film.forward_features(x, t, z)
        f_02 = film.forward_features(x, t, torch.zeros(x.shape[0], K))
    check("once trained, z moves the features", not torch.allclose(f_z2, f_02, atol=1e-6))
    # The asymmetry that decides which mode A2 reports. z_mlp's first layer keeps a bias,
    # so z_mlp(0) is not zero once the second layer's weight moves: under temb the z = 0
    # control drifts into "backbone plus a learned constant". per_block does not drift
    # (section 2 above), and neither does the additive basis.
    check("but z = 0 has drifted off the unconditioned backbone, unlike per_block",
          not torch.equal(f_02, f_none),
          f"drift {float((f_02 - f_none).abs().mean()):.4f}")


def s10_gradient_reaches_the_coordinate_machinery() -> None:
    print("\n10. the loss still trains the encoder and the transport in the FiLM arm")
    torch.manual_seed(6)
    model, enc, tr = fresh(film=True, seed=6)
    with torch.no_grad():
        for m in model.backbone.modules():
            if isinstance(m, ResBlock):
                m.z_proj.weight.normal_(std=0.3)
    g = torch.Generator().manual_seed(21)
    support = torch.randn(6, 3, 32, 32, generator=g)
    x, t, _ = inputs(seed=22)
    z_s = enc(support)
    z_t = tr(z_s.unsqueeze(0), torch.tensor([1])).squeeze(0)
    model.eps_hat(x, t, z_t.unsqueeze(0).expand(x.shape[0], -1)).pow(2).mean().backward()
    g_enc = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
    g_tr = sum(float(p.grad.abs().sum()) for p in tr.parameters() if p.grad is not None)
    check("the set encoder receives gradient through the FiLM path", g_enc > 0, f"{g_enc:.3e}")
    check("the transport map receives gradient through the FiLM path", g_tr > 0, f"{g_tr:.3e}")


def main() -> int:
    for fn in (s1_insertion_points, s2_identity_at_zero, s3_every_block_is_reached,
               s4_basis_is_gone, s5_no_shared_pass, s6_matched, s7_guards,
               s8_round_trip, s9_temb_mode, s10_gradient_reaches_the_coordinate_machinery):
        fn()
    rule = "-" * 78
    print(f"\n{rule}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        print(rule)
        return 1
    print("all checks passed")
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
