"""Controlled GMM boundary analysis for Meta-Diff (engineer note).

Phase diagram over source-task identifiability (alpha) x relation heterogeneity (eta), plus a
separate transport-nonlinearity (kappa) sweep. Reuses the stage-1 GMM Meta-Diff implementation
unchanged; only component *means* are manipulated. See the note for the exact formulas.

  python -m runner.gmm_boundary --smoke          # cells (0,0),(1,0),(1,2), verify note section 11
  python -m runner.gmm_boundary --cell A E --seed S --steps N --out row.json   # one grid cell
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np, torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.gmm2d import GMM2D, random_gmm, GMMTask
from episodes.gmm_episodes import GMMEpisodeLoader, as_batch, as_points
from evaluation.metrics_analytic import sliced_wasserstein, energy_mmd
from config.base_config import BaseConfig
from diffusion.schedule import NoiseSchedule
from diffusion.sampler import ddim_sample, make_eps_fn
from models.backbone import build_backbone
from models.score_model import ScoreModel
from models.set_encoder import VectorSetEncoder
from models.transport import Transport
from training.meta_train import meta_step
from adaptation.budget import AdaptBudget
from adaptation.coordinate import adapt
import models.mlp_backbone  # noqa: F401 register mlp_vector

ALPHAS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50]
ETAS = [0.0, 0.10, 0.25, 0.50, 1.00, 2.00]
_R25 = math.radians(25.0)
R_MAT = 1.10 * torch.tensor([[math.cos(_R25), -math.sin(_R25)],
                             [math.sin(_R25), math.cos(_R25)]], dtype=torch.float32)
B_VEC = torch.tensor([0.75, -0.50], dtype=torch.float32)


def T0(mu):  # mu (...,2) -> (...,2):  R mu + b
    return mu @ R_MAT.T.to(mu) + B_VEC.to(mu)


# ---------------- base pool + fixed perturbations (fixed before training) ----------------
def base_pool(seed, ntr=64, nva=16, nte=64, J=4, spread=2.5):
    """Per-task random MEANS only; mixture weights and covariances are frozen to shared
    canonical values (note section 2), so alpha=0 makes source tasks truly identical."""
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    canon = random_gmm(J, generator=g)              # canonical weights + covariances (shared)
    W, C = canon.weights, canon.covs
    def task_gmm():
        means = (torch.rand(J, 2, generator=g) * 2 - 1) * spread
        return GMM2D(W.clone(), means, C.clone())
    gmms = [task_gmm() for _ in range(ntr + nva + nte)]
    return {"train": gmms[:ntr], "val": gmms[ntr:ntr + nva], "test": gmms[ntr + nva:]}, J


def xi_draws(seed, pool, J):
    g = torch.Generator(device="cpu").manual_seed(2000 + seed)
    xi = {sp: torch.randn(len(pool[sp]), J, 2, generator=g) for sp in ("train", "val", "test")}
    rms = xi["train"].pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)   # unit-RMS over train
    return {sp: v / rms for sp, v in xi.items()}


def make_cell(pool, xi, J, alpha, eta):
    """Build source/target GMMs for one (alpha, eta) cell. Means only; weights/covs fixed."""
    tr_means = torch.stack([g.means for g in pool["train"]])          # (Ntr,J,2)
    mu_bar = tr_means.mean(0)                                         # (J,2), train only
    def muS(g): return mu_bar + alpha * (g.means - mu_bar)
    trS = torch.stack([muS(g) for g in pool["train"]])
    sR = (T0(trS) - trS).pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)  # RMS over train
    tasks, tid = {}, 0
    for sp in ("train", "val", "test"):
        lst = []
        for i, g in enumerate(pool[sp]):
            mS = muS(g)
            mT = T0(mS) + eta * sR * xi[sp][i]
            src = GMM2D(g.weights.clone(), mS, g.covs.clone())
            tgt = GMM2D(g.weights.clone(), mT, g.covs.clone())
            lst.append(GMMTask(tid, "affine", 0, src, tgt)); tid += 1
        tasks[sp] = lst
    return tasks, {"mu_bar": mu_bar, "sR": float(sR)}


# ---------------- independent diagnostics (must not use Meta-Diff) ----------------
def _gmm_logprob(gmm: GMM2D, x):  # clean log p(x); x (n,2) -> (n,)
    diff = x.unsqueeze(1) - gmm.means.unsqueeze(0)                    # (n,J,2)
    Sinv = torch.linalg.inv(gmm.covs)
    maha = torch.einsum("njd,jde,nje->nj", diff, Sinv, diff)
    logdet = torch.logdet(gmm.covs)
    logn = -0.5 * (maha + logdet + 2 * math.log(2 * math.pi))
    return torch.logsumexp(gmm.weights.log().unsqueeze(0) + logn, dim=1)


def diag_bayes_id(sources, MS, gen):
    """Top-1 Bayes task-ID over held-out test sources using MS source samples each."""
    n = len(sources); correct = 0
    for i, gi in enumerate(sources):
        x = gi.sample(MS, gen)
        ll = torch.tensor([float(_gmm_logprob(gj, x).sum()) for gj in sources])
        correct += int(ll.argmax().item() == i)
    return correct / n


def diag_pairwise_sw(sources, n_samp, gen, proj):
    S = [g.sample(n_samp, gen) for g in sources]
    d, cnt = 0.0, 0
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d += _sw_proj(S[i], S[j], proj); cnt += 1
    return d / max(1, cnt)


def _sw_proj(a, b, proj):
    pa = (a @ proj).sort(0).values; pb = (b @ proj).sort(0).values
    n = min(pa.shape[0], pb.shape[0])
    ia = (torch.linspace(0, pa.shape[0] - 1, n)).long(); ib = (torch.linspace(0, pb.shape[0] - 1, n)).long()
    return float((pa[ia] - pb[ib]).pow(2).mean().sqrt())


def diag_oracle_r2(tasks):
    """Oracle affine fit source-means->target-means on train components; held-out R2 on test."""
    def XY(sp):
        X = torch.cat([t.source.means for t in tasks[sp]])           # (n*J,2)
        Y = torch.cat([t.target.means for t in tasks[sp]])
        return X, Y
    Xtr, Ytr = XY("train"); Xte, Yte = XY("test")
    Xa = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)            # affine
    W = torch.linalg.lstsq(Xa, Ytr).solution                        # (3,2)
    Xte_a = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
    pred = Xte_a @ W
    ss_res = (Yte - pred).pow(2).sum(); ss_tot = (Yte - Yte.mean(0)).pow(2).sum()
    rmse = (Yte - pred).pow(2).mean().sqrt()
    return float(1 - ss_res / ss_tot.clamp_min(1e-8)), float(rmse)


@torch.no_grad()
def diag_learned_id(enc, test_tasks_dev, ms, dev, gen, n_ref=8, n_query=8):
    """Learned-space task identifiability: nearest-centroid task-ID in the trained set-encoder's
    z_S space, same MS-source protocol as the Bayes diagnostic. High Bayes-ID but low learned-ID
    means the encoder collapses recoverable task identity (a representation failure), as opposed
    to the task being intrinsically unidentifiable (both low)."""
    cents = []
    for t in test_tasks_dev:
        zs = torch.stack([enc(as_batch(t.source.sample(ms, gen).to(dev))) for _ in range(n_ref)])
        cents.append(zs.mean(0))
    C = torch.stack(cents)                                          # (T, k)
    correct = total = 0
    for i, t in enumerate(test_tasks_dev):
        for _ in range(n_query):
            zq = enc(as_batch(t.source.sample(ms, gen).to(dev)))
            pred = int((C - zq).pow(2).sum(1).argmin().item())
            correct += (pred == i); total += 1
    return correct / max(1, total)


# ---------------- model build / train / eval one cell ----------------
def build_cell(cfg, dev, transport_kind="linear"):
    sched = NoiseSchedule(cfg.diffusion.n_steps, cfg.diffusion.schedule)
    bb = build_backbone("mlp_vector", image_channels=2, hidden=256, depth=4, feature_channels=128)
    model = ScoreModel(bb, sched, k=cfg.model.k, coord_decoder="linear").to(dev)
    enc = VectorSetEncoder(dim=2, feature_dim=128, k=cfg.model.k, hidden=128).to(dev)
    tr = Transport(k=cfg.model.k, n_relations=None, kind=transport_kind).to(dev)
    return model, enc, tr


def train_cell(model, enc, tr, tasks_train, cfg, steps, dev, ms=32):
    loader = GMMEpisodeLoader(tasks_train, dev, m_source=ms, query_batch=128,
                              k_shots=(1, 5), seed=0)
    params = [p for m in (model, enc, tr) for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr)
    for m in (model, enc, tr): m.train()
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = cfg.train.lr * min(1.0, step / 200)
        out = meta_step(model, enc, tr, loader.sample(), cfg)
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip); opt.step()
    for m in (model, enc, tr): m.eval()


@torch.no_grad()
def mean_source_coord(enc, tasks_train, dev, ms, gen):
    zs = []
    for t in tasks_train:
        x = as_batch(t.source.sample(ms, gen).to(dev))
        zs.append(enc(x))
    return torch.stack(zs).mean(0)


@torch.no_grad()
def gen_sw(model, z, target_gmm, n_gen, n_real, dev, gen):
    fake = as_points(ddim_sample(model.schedule, make_eps_fn(model, z), (n_gen, 2, 1, 1),
                                 dev, n_steps=50, generator=gen, clip_x0=8.0))
    real = target_gmm.sample(n_real, gen)
    return sliced_wasserstein(fake, real, generator=gen), energy_mmd(fake[:1024], real[:1024])


@torch.no_grad()
def eval_cell(model, enc, tr, tasks, cfg, dev, kt=0, n_gen=4096, n_real=10000,
              n_tasks=None, ms=32, wrong_shift=1):
    gen = torch.Generator(device=dev).manual_seed(0)
    test = [t.to(dev) for t in (tasks["test"][:n_tasks] if n_tasks else tasks["test"])]
    zbar = mean_source_coord(enc, [t.to(dev) for t in tasks["train"]], dev, ms, gen)
    rows = []
    for i, t in enumerate(test):
        zS = enc(as_batch(t.source.sample(ms, gen).to(dev)))
        wrong = test[(i + wrong_shift) % len(test)]
        zS_wrong = enc(as_batch(wrong.source.sample(ms, gen).to(dev)))
        methods = {"correct": tr(zS), "wrong": tr(zS_wrong), "relation_only": tr(zbar),
                   "source_reuse": zS}
        if kt > 0:
            methods["target_only"] = enc(as_batch(t.target.sample(kt, gen).to(dev)))
        methods["oracle"] = enc(as_batch(t.target.sample(512, gen).to(dev)))
        names = list(methods)
        M = len(names)
        zbig = torch.stack([methods[n] for n in names]).repeat_interleave(n_gen, 0)   # (M*n_gen,k)
        x_init = torch.randn(n_gen, 2, 1, 1, device=dev, generator=gen).repeat(M, 1, 1, 1)  # paired noise
        def eps_fn(x_t, tt, _z=zbig):
            return model.eps_hat(x_t, tt, _z)
        fake_all = as_points(ddim_sample(model.schedule, eps_fn, (M * n_gen, 2, 1, 1), dev,
                                         n_steps=50, generator=gen, clip_x0=8.0, x_init=x_init))
        real = t.target.sample(n_real, gen)
        r = {"task": t.task_id}
        for mi, name in enumerate(names):
            fk = fake_all[mi * n_gen:(mi + 1) * n_gen]
            r[f"SW_{name}"] = sliced_wasserstein(fk, real, generator=gen)
            r[f"MMD_{name}"] = energy_mmd(fk[:1024], real[:1024])
        r["G_source"] = r["SW_wrong"] - r["SW_correct"]
        r["G_vs_relation"] = r["SW_relation_only"] - r["SW_correct"]
        if kt > 0:
            r["G_transfer"] = r["SW_target_only"] - r["SW_correct"]
        rows.append(r)
    return rows


def run_cell(alpha, eta, seed, steps, dev, cfg, n_tasks=None, n_gen=4096, n_real=10000,
             ms=32, transport_kind="linear", kt=0):
    pool, J = base_pool(seed)
    xi = xi_draws(seed, pool, J)
    tasks, meta = make_cell(pool, xi, J, alpha, eta)
    proj = torch.randn(2, 256, generator=torch.Generator(device="cpu").manual_seed(7))
    proj = proj / proj.norm(0, keepdim=True)
    gcpu = torch.Generator(device="cpu").manual_seed(seed)
    src_test = [t.source for t in tasks["test"][:(n_tasks or len(tasks["test"]))]]
    diags = {"bayes_id": diag_bayes_id(src_test, ms, gcpu),
             "pairwise_sw": diag_pairwise_sw(src_test, 2000 if n_tasks else 10000, gcpu, proj),
             "oracle_r2": diag_oracle_r2(tasks)[0], "oracle_rmse": diag_oracle_r2(tasks)[1]}
    model, enc, tr = build_cell(cfg, dev, transport_kind)
    train_cell(model, enc, tr, tasks["train"], cfg, steps, dev, ms)
    # learned-space task identifiability (needs the trained encoder): same MS-source
    # protocol and same test tasks as diag_bayes_id, so the two accuracies are directly
    # comparable and share the 1/n_test chance level.
    lid_test = [t.to(dev) for t in tasks["test"][:(n_tasks or len(tasks["test"]))]]
    lid_gen = torch.Generator(device=dev).manual_seed(1234 + seed)
    diags["learned_id"] = diag_learned_id(enc, lid_test, ms, dev, lid_gen)
    diags["learned_id_chance"] = 1.0 / len(lid_test)
    rows = eval_cell(model, enc, tr, tasks, cfg, dev, kt=kt, n_gen=n_gen, n_real=n_real,
                     n_tasks=n_tasks, ms=ms)
    G = float(np.mean([r["G_source"] for r in rows]))
    return {"alpha": alpha, "eta": eta, "seed": seed, "steps": steps, "kt": kt,
            "transport": transport_kind, "diagnostics": diags,
            "G_source_mean": G, "rows": rows, "cell_meta": {"sR": meta["sR"]}}


def smoke(dev, cfg):
    print("=== GMM boundary smoke (note section 11) ===")
    res = {}
    for (a, e) in [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]:
        t0 = time.time()
        r = run_cell(a, e, seed=0, steps=1500, dev=dev, cfg=cfg, n_tasks=16, n_gen=512,
                     n_real=2000, ms=32)
        res[(a, e)] = r
        d = r["diagnostics"]
        print(f"  cell (a={a},e={e}) [{time.time()-t0:.0f}s]: bayes_id={d['bayes_id']:.3f} "
              f"learned_id={d['learned_id']:.3f} (chance={d['learned_id_chance']:.3f}) "
              f"pairwise_sw={d['pairwise_sw']:.3f} oracle_r2={d['oracle_r2']:.3f} "
              f"G_source={r['G_source_mean']:+.4f}")
    print("\n-- note section 11 checks --")
    c1 = res[(0.0, 0.0)]["diagnostics"]["bayes_id"]
    print(f"(i)  source-ID chance at alpha=0: {c1:.3f} (chance=1/16={1/16:.3f})  -> {'OK' if c1 < 0.15 else 'CHECK'}")
    g10 = res[(1.0, 0.0)]["G_source_mean"]
    print(f"(ii) correct-source advantage positive at (1,0): G_source={g10:+.4f} -> {'OK' if g10 > 0 else 'CHECK'}")
    r10 = res[(1.0, 0.0)]["diagnostics"]["oracle_r2"]; r12 = res[(1.0, 2.0)]["diagnostics"]["oracle_r2"]
    print(f"(iii) oracle relation R2 falls with eta: R2(1,0)={r10:.3f} > R2(1,2)={r12:.3f} -> {'OK' if r10 > r12 else 'CHECK'}")
    l0 = res[(0.0, 0.0)]["diagnostics"]["learned_id"]; l1 = res[(1.0, 0.0)]["diagnostics"]["learned_id"]
    b0 = res[(0.0, 0.0)]["diagnostics"]["bayes_id"]; b1 = res[(1.0, 0.0)]["diagnostics"]["bayes_id"]
    print(f"(iv) learned-ID tracks identifiability: learned_id(0,0)={l0:.3f} (bayes {b0:.3f}) < "
          f"learned_id(1,0)={l1:.3f} (bayes {b1:.3f}) -> {'OK' if l1 > l0 else 'CHECK'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell", nargs=2, type=float, default=None, metavar=("ALPHA", "ETA"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--kt", type=int, default=0)
    ap.add_argument("--transport", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg = BaseConfig(); cfg.model.k = 16; cfg.model.n_relations = None; cfg.train.lr = 2e-3
    if a.smoke:
        smoke(dev, cfg); return
    r = run_cell(a.cell[0], a.cell[1], a.seed, a.steps, dev, cfg, kt=a.kt, transport_kind=a.transport)
    print(f"cell a={a.cell[0]} e={a.cell[1]} seed={a.seed}: G_source={r['G_source_mean']:+.4f} "
          f"bayes_id={r['diagnostics']['bayes_id']:.3f} oracle_r2={r['diagnostics']['oracle_r2']:.3f}")
    if a.out:
        json.dump(r, open(a.out, "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
