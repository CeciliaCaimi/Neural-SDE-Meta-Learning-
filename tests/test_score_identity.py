"""Mathematical identities of the meta layer.

    python -m tests.test_score_identity

These are the relations the document requires to hold exactly:
  - x_t = alpha_t x_0 + sigma_t eps,  with alpha_t^2 + sigma_t^2 = 1
  - eps*(x_t,t) = -sigma_t s*(x_t,t)                              equation (19)
  - eps_hat_{phi,z} = eps_hat_{phi,0} + sum_l z_l R_{phi,l}       equation (21)
      - reduces to eps_hat_0 at z=0 (the basis of the z=0 control)
      - strictly linear in z (the basis is additive)
      - a one-hot z selects exactly the l-th basis direction
  - z receives gradient (the basis head init must not pin dL/dz at 0)
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusion.forward import from_model_output, q_sample, to_model_input   # noqa: E402
from diffusion.losses import denoising_loss                                 # noqa: E402
from diffusion.schedule import NoiseSchedule                                # noqa: E402
from models.score_model import ScoreModel                                   # noqa: E402
from models.unet import SmallUNet                                           # noqa: E402

RULE = "─" * 78
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ——  {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)
    print(RULE)
    print("meta-layer identity verification")
    print(RULE)

    B, C, S, K = 4, 3, 32, 8
    sched = NoiseSchedule(n_steps=1000, kind="cosine")
    model = ScoreModel(SmallUNet(image_channels=C, image_size=S), sched, k=K).eval()

    n = model.n_parameters()
    print(f"backbone {type(model.backbone).__name__} | k={K} | "
          f"phi has {n['total_phi']/1e6:.2f}M parameters | {n['adapted_at_deployment']} scalars move at deployment\n")

    # ---- 1. schedule ----
    print("[1] noise schedule")
    a2s2 = (sched.alpha ** 2 + sched.sigma ** 2)
    record("α_t² + σ_t² = 1", torch.allclose(a2s2, torch.ones_like(a2s2), atol=1e-5),
           f"max deviation {(a2s2 - 1).abs().max().item():.2e}")
    record("sigma_t increases monotonically", bool((sched.sigma[1:] >= sched.sigma[:-1]).all()),
           f"σ ∈ [{sched.sigma[0]:.4f}, {sched.sigma[-1]:.4f}]")

    # ---- 2. Forward noising ----
    print("\n[2] forward noising q(x_t | x_0)")
    x0 = torch.randn(B, C, S, S)
    t = torch.tensor([0, 250, 500, 999])
    eps = torch.randn_like(x0)
    nb = q_sample(sched, x0, t, eps)
    want = sched.alpha_t(t, 4) * x0 + sched.sigma_t(t, 4) * eps
    record("x_t = alpha_t x_0 + sigma_t eps", torch.allclose(nb.x_t, want, atol=0, rtol=0), "bit-for-bit equal")

    u8 = torch.randint(0, 256, (B, C, S, S), dtype=torch.uint8)
    record("uint8 <-> [-1,1] round trip is lossless", bool((from_model_output(to_model_input(u8.clone())) == u8).all()))

    # ---- 3. Equation (19) ----
    print("\n[3] equation (19)  eps* = -sigma_t s*")
    z = torch.randn(B, K)
    with torch.no_grad():
        e = model.eps_hat(nb.x_t, t, z)
        s = model.score(nb.x_t, t, z)
    record("score(x,t,z) = −ε̂/σ_t",
           torch.allclose(s, -e / sched.sigma_t(t, 4), atol=1e-6),
           f"max deviation {(s + e / sched.sigma_t(t, 4)).abs().max().item():.2e}")
    record("eps <-> score round trip is consistent",
           torch.allclose(sched.score_to_eps(sched.eps_to_score(e, t), t), e, atol=1e-5))

    # ---- 4. Equation (21) ----
    print("\n[4] equation (21)  eps_hat_z = eps_hat_0 + sum z_l R_l")
    with torch.no_grad():
        feats = model.features(nb.x_t, t)
        base = model.eps_base(nb.x_t, t, feats)
        R = model.basis(nb.x_t, t, feats)
        e_zero = model.eps_hat(nb.x_t, t, None)

    record("basis shape (B,k,C,H,W)", tuple(R.shape) == (B, K, C, S, S), str(tuple(R.shape)))
    record("eps_hat_z equals eps_hat_0 at z=0", torch.allclose(e_zero, base, atol=0, rtol=0), "bit-for-bit equal")

    z1, z2 = torch.randn(B, K), torch.randn(B, K)
    with torch.no_grad():
        r1 = model.eps_hat(nb.x_t, t, z1) - base
        r2 = model.eps_hat(nb.x_t, t, z2) - base
        r12 = model.eps_hat(nb.x_t, t, z1 + z2) - base
    record("the residual is strictly linear in z", torch.allclose(r12, r1 + r2, atol=1e-5),
           f"max deviation {(r12 - r1 - r2).abs().max().item():.2e}")

    ell = 3
    onehot = torch.zeros(B, K)
    onehot[:, ell] = 1.0
    with torch.no_grad():
        r_hot = model.eps_hat(nb.x_t, t, onehot) - base
    record(f"a one-hot z selects basis direction {ell}",
           torch.allclose(r_hot, R[:, ell], atol=1e-6))

    # ---- 5. Diagnostic quantities ----
    print("\n[5] the r_basis diagnostic (section 15)")
    with torch.no_grad():
        r0 = model.basis_usage(nb.x_t, t, torch.zeros(B, K))
        rz = model.basis_usage(nb.x_t, t, torch.randn(B, K))
    record("r_basis = 0 when z = 0", bool((r0 == 0).all()))
    record("r_basis > 0 for a random z", bool((rz > 0).all()),
           f"r_basis is about {rz.mean().item():.2e} after init; the small basis-head init should rise during training")

    # ---- 6. Gradients ----
    print("\n[6] gradient reachability")
    model.train()
    z_leaf = torch.randn(B, K, requires_grad=True)
    pred = model.eps_hat(nb.x_t, t, z_leaf)
    loss = denoising_loss(nb.eps, pred, t, sched)
    loss.backward()
    gz = z_leaf.grad
    record("dL/dz is nonzero", gz is not None and bool((gz.abs() > 0).any()),
           f"||dL/dz|| = {gz.norm().item():.3e}  (if 0, the encoder receives no basis gradient)")
    record("basis_head receives gradient", bool(model.basis_head.weight.grad.abs().sum() > 0),
           f"‖grad‖ = {model.basis_head.weight.grad.norm().item():.3e}")
    record("the backbone receives gradient",
           bool(model.backbone.conv_in.weight.grad.abs().sum() > 0))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(RULE)
    print(f"{len(results) - n_fail} / {len(results)} checks passed" + ("" if not n_fail else f"  --  {n_fail} failed"))
    print(RULE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
