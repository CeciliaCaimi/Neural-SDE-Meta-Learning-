"""C2 for ChestX-ray14: frozen evaluation predictors, trained once and loaded everywhere.

The plan asks, before any headline number, for "a frozen semantic / diagnostic consistency
predictor, and a demographic or phenotype consistency predictor", each reporting its own
held-out ceiling beside every verdict it is used for. Fitzpatrick had a fixed statistic for its
population axis -- the individual typology angle, monotone across all six skin types -- and it
does not transfer: it measures skin pigmentation. There is no standard fixed statistic for age
on a chest film, so here all three instruments are trained, which makes the ceilings the thing
every reading has to be judged against.

Three instruments, one per question a generated set has to answer:

``age``      a regressor. Did generation move to the target *age group*? Read as the mean
             predicted age of a set, compared with the same instrument's reading of the real
             held-out set -- so the instrument's own bias cancels in the difference.
``finding``  a classifier over the findings. Is the generated film still the same *disease*?
``view``     AP against PA. Not a question the method is asked, but the confound the audit
             found: the AP share differs by 14.7 points between the source bin and the target
             bins, so a model could move towards a target group by changing the imaging
             geometry rather than the patient. Every verdict reports it beside the age reading.

**Trained once, frozen, saved.** "Frozen" in the plan is taken literally: the instruments are
trained a single time with a fixed seed, written to a file with the split checksum inside it,
and every later evaluation -- C3, the fine-tuning budget selection, C4 and the FiLM arm --
loads that file. Retraining per run would make two verdicts rest on two instruments.

**Leakage is excluded by patient, not by image.** Every patient appearing in any held-out
target query -- the real reference sets a generated set is compared against -- is removed from
the instruments' training data, and the ceiling is measured on a further held-out 15 % of
patients. Source pools and K_T supports may be seen: they are inputs to every method, not the
reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from domains.chestxray import FINDINGS, ZipIndex, load_annotations, load_chestxray
from evaluation.instruments import TinyCNN


@dataclass
class CXRInstruments:
    age: TinyCNN
    view: TinyCNN
    finding: TinyCNN
    findings: list[str]
    age_mean: float
    age_std: float
    report: dict = field(default_factory=dict)

    # ---- readings -----------------------------------------------------------------

    @torch.no_grad()
    def predicted_age(self, x: Tensor) -> Tensor:
        return self.age(x).squeeze(1) * self.age_std + self.age_mean

    @torch.no_grad()
    def ap_share(self, x: Tensor) -> float:
        return float((self.view(x).argmax(1) == 1).float().mean())

    @torch.no_grad()
    def finding_share(self, x: Tensor, finding: str) -> float:
        return float((self.finding(x).argmax(1) == self.findings.index(finding)).float().mean())

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        return self.finding.features(x)

    def eval(self) -> "CXRInstruments":
        for m in (self.age, self.view, self.finding):
            m.eval()
        return self


def _to_float(images: np.ndarray, idx: np.ndarray, device) -> Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device)
    return x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def _fit(model, x: Tensor, y: Tensor, loss_fn, *, epochs: int, batch: int, lr: float,
         rng: np.random.Generator) -> int:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, x.shape[0] // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    model.train()
    done = 0
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(x.shape[0])).to(x.device)
        for s in range(0, x.shape[0] - batch + 1, batch):
            sel = perm[s:s + batch]
            xb, yb = x.index_select(0, sel), y.index_select(0, sel)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    model.eval()
    return done


@torch.no_grad()
def _predict(model, x: Tensor, batch: int = 512) -> Tensor:
    return torch.cat([model(x[s:s + batch]) for s in range(0, x.shape[0], batch)])


def build_instruments(split: dict, *, device, seed: int = 4321, cap: int = 40000,
                      edges: list[int] | None = None, verbose: bool = True
                      ) -> CXRInstruments:
    """Train the three instruments on patients that appear in no held-out query."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ann = load_annotations()
    edges = edges or split["config"]["edges"]

    # every patient in any held-out query stream, of any split, is out of bounds
    query_pat = {ann[n].patient for t in split["tasks"].values()
                 for tt in t["targets"].values() for n in tt["tgt_query"]}
    eligible = [a for a in ann.values() if a.patient not in query_pat and 0 <= a.age < 100
                and a.view in ("AP", "PA")]
    # patient-level hold-out for the ceiling
    pats = sorted({a.patient for a in eligible})
    hold_p = set(pats[i] for i in rng.permutation(len(pats))[:int(0.15 * len(pats))])
    pool = [a for a in eligible if a.patient not in hold_p]
    held = [a for a in eligible if a.patient in hold_p]
    if len(pool) > cap:
        pool = [pool[i] for i in rng.permutation(len(pool))[:cap]]
    if len(held) > cap // 4:
        held = [held[i] for i in rng.permutation(len(held))[:cap // 4]]
    if verbose:
        print(f"  instruments: {len(query_pat)} held-out-query patients excluded; training on "
              f"{len(pool)} images, ceiling measured on {len(held)} images of "
              f"{len(hold_p)} further held-out patients")

    idx = ZipIndex()
    names = sorted({a.filename for a in pool} | {a.filename for a in held})
    raw = load_chestxray(names=names, index=idx, verbose=verbose)
    ix = {n: i for i, n in enumerate(raw.names)}
    tr_i = np.array([ix[a.filename] for a in pool])
    ho_i = np.array([ix[a.filename] for a in held])
    x_tr, x_ho = _to_float(raw.images, tr_i, device), _to_float(raw.images, ho_i, device)

    # ---- age: a regressor on standardised age -----------------------------------------
    ages_tr = np.array([a.age for a in pool], dtype=np.float32)
    mu, sd = float(ages_tr.mean()), float(ages_tr.std())
    y_age = torch.from_numpy((ages_tr - mu) / sd).to(device).unsqueeze(1)
    age = TinyCNN(1).to(device)
    _fit(age, x_tr, y_age, F.mse_loss, epochs=10, batch=256, lr=2e-3, rng=rng)

    # ---- view: AP against PA -----------------------------------------------------------
    y_view = torch.tensor([1 if a.view == "AP" else 0 for a in pool], device=device)
    view = TinyCNN(2).to(device)
    _fit(view, x_tr, y_view, F.cross_entropy, epochs=6, batch=256, lr=2e-3, rng=rng)

    # ---- finding: single-label films of the fourteen findings --------------------------
    fl = [(i, a) for i, a in zip(tr_i, pool) if a.finding in FINDINGS]
    findings = sorted({a.finding for _, a in fl})
    row = {f: k for k, f in enumerate(findings)}
    x_f = _to_float(raw.images, np.array([i for i, _ in fl]), device)
    y_f = torch.tensor([row[a.finding] for _, a in fl], device=device)
    finding = TinyCNN(len(findings)).to(device)
    _fit(finding, x_f, y_f, F.cross_entropy, epochs=12, batch=256, lr=2e-3, rng=rng)

    inst = CXRInstruments(age=age, view=view, finding=finding, findings=findings,
                          age_mean=mu, age_std=sd).eval()

    # ---- ceilings, on the held-out patients ---------------------------------------------
    true_age = np.array([a.age for a in held], dtype=np.float32)
    pred_age = inst.predicted_age(x_ho).cpu().numpy() if len(held) else np.zeros(0)
    true_view = np.array([a.view for a in held])
    view_pred = _predict(view, x_ho).argmax(1).cpu().numpy()

    def binof(a: float) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= a < edges[i + 1]:
                return i
        return -1

    by_bin = {}
    for b in range(len(edges) - 1):
        for v in ("all", "PA", "AP"):
            m = np.array([binof(t) == b for t in true_age])
            if v != "all":
                m &= true_view == v
            if m.sum() >= 5:
                by_bin[f"{edges[b]}-{edges[b+1]}|{v}"] = (float(pred_age[m].mean()), int(m.sum()))
    # the confound read directly: same true age bin, different view
    fh = [(k, a) for k, a in enumerate(held) if a.finding in row]
    fpred = _predict(finding, x_ho[[k for k, _ in fh]]).argmax(1).cpu().numpy() if fh else []
    per_finding = {}
    for f in findings:
        ks = [j for j, (_, a) in enumerate(fh) if a.finding == f]
        if ks:
            per_finding[f] = (float(np.mean([fpred[j] == row[f] for j in ks])), len(ks))

    inst.report = {
        "n_train": len(pool), "n_heldout": len(held),
        "excluded_query_patients": len(query_pat),
        "age_mae": float(np.abs(pred_age - true_age).mean()),
        "age_r": float(np.corrcoef(pred_age, true_age)[0, 1]),
        "age_sd_true": float(true_age.std()),
        "age_by_bin_and_view": by_bin,
        "view_acc": float((view_pred == (true_view == "AP")).mean()),
        "view_chance": float(max((true_view == "AP").mean(), (true_view == "PA").mean())),
        "finding_acc": float(np.mean([fpred[j] == row[a.finding] for j, (_, a) in enumerate(fh)]))
        if fh else float("nan"),
        "finding_chance": 1.0 / len(findings),
        "finding_per_class": per_finding,
        "seed": seed, "split_checksum": split.get("checksum", ""),
    }
    return inst


def save_instruments(path: str, inst: CXRInstruments) -> None:
    torch.save({"age": inst.age.state_dict(), "view": inst.view.state_dict(),
                "finding": inst.finding.state_dict(), "findings": inst.findings,
                "age_mean": inst.age_mean, "age_std": inst.age_std,
                "report": inst.report}, path)


def load_instruments(path: str, device, split_checksum: str | None = None) -> CXRInstruments:
    """Load the frozen instruments. Refuses a file built against a different split: its
    exclusion of held-out query patients would then be the wrong exclusion."""
    sd = torch.load(path, map_location=device, weights_only=False)
    if split_checksum is not None and sd["report"].get("split_checksum") != split_checksum:
        raise RuntimeError(
            f"{path} was built for split {sd['report'].get('split_checksum', '?')[:12]}, not "
            f"{split_checksum[:12]}; its leakage exclusion does not hold for this split. "
            "Rebuild it with scripts/cxr_c2_instruments.py.")
    age, view = TinyCNN(1).to(device), TinyCNN(2).to(device)
    finding = TinyCNN(len(sd["findings"])).to(device)
    age.load_state_dict(sd["age"]); view.load_state_dict(sd["view"])
    finding.load_state_dict(sd["finding"])
    return CXRInstruments(age=age, view=view, finding=finding, findings=sd["findings"],
                          age_mean=sd["age_mean"], age_std=sd["age_std"],
                          report=sd["report"]).eval()


def ceiling_lines(inst: CXRInstruments) -> list[str]:
    """What the instruments are worth, in a form every results file prints first."""
    r = inst.report
    out = [f"frozen instruments (seed {r['seed']}, split {r['split_checksum'][:12]}...): "
           f"trained on {r['n_train']} films, ceilings on {r['n_heldout']} films of "
           f"held-out patients; {r['excluded_query_patients']} held-out-query patients excluded",
           f"  age      MAE {r['age_mae']:.1f} years, r = {r['age_r']:.3f} "
           f"(true age sd {r['age_sd_true']:.1f})",
           f"  view     {100*r['view_acc']:.1f}% (majority class {100*r['view_chance']:.1f}%)",
           f"  finding  {100*r['finding_acc']:.1f}% over {len(inst.findings)} findings "
           f"(chance {100*r['finding_chance']:.1f}%)"]
    return out
