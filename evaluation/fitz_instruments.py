"""C2: the frozen evaluation predictors for Stage C.

On CIFAR the headline instrument was ``transform_signature`` -- three fixed convolutions
that read blur, added noise and contrast reduction, with nothing in them that can drift. No
such thing exists for a phototype shift, so **these trained predictors are the instrument the
whole Stage C result rests on**, which is why C2 comes before any headline number and why each
reports its own held-out ceiling beside every verdict it is used for.

Two predictors, answering the two questions a generated set has to pass:

``train_phenotype_instrument``
    Did generation move to the *target population*? A binary Fitzpatrick I-III against IV-VI
    classifier. It is trained only on images of **training and validation** conditions, so it
    never sees a test condition, and its accuracy is reported twice: on held-out images of the
    conditions it trained on, and on test-condition images, which is the transfer it is
    actually used for. The second number is the one that bounds a verdict.

``train_condition_instrument``
    Is the generated image still the *same disease*? An N-way condition classifier over every
    condition with enough images. Two things make it weak and both are stated rather than
    hidden: 32x32 is a poor resolution for a lesion, and the reachable archive is a quarter of
    the dataset. Its ceiling is printed with the chance rate beside it, and a ceiling near
    chance means the semantic reading is uninformative -- not that the arms are equal.

Leakage is excluded by construction, not by hope: every hash the caller names in ``exclude``
is dropped from training, and the caller passes the split's held-out target query streams,
which are the real reference sets a generated set is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from domains.fitzpatrick import SOURCE_PHOTOTYPES, TARGET_PHOTOTYPES, FitzRaw
from evaluation.instruments import TinyCNN


# ---------------------------------------------------------------------------
# The fixed population statistic: no training, nothing that can drift
# ---------------------------------------------------------------------------
#
# This is Stage C's transform_signature. Skin pigmentation has a standard objective measure
# -- the individual typology angle, ITA, computed in CIELAB and used in dermatology precisely
# to place a subject on the Fitzpatrick scale:
#
#     ITA = atan2(L* - 50, b*) * 180 / pi        higher = lighter
#
# Nothing here is learned, so unlike the trained predictors below it cannot drift, cannot
# overfit and carries no ceiling of its own. It is the headline population reading and the
# classifier is the secondary one, which is the same arrangement Stage A used.

# sRGB (D65) to XYZ, and the D65 white point.
_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_WHITE = (0.95047, 1.00000, 1.08883)


def _srgb_to_lab(x: Tensor) -> Tensor:
    """(B, 3, H, W) in [-1, 1] -> (B, 3, H, W) as L*, a*, b*."""
    c = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp_min(0) ** 2.4)
    m = torch.tensor(_RGB_TO_XYZ, dtype=lin.dtype, device=lin.device)
    xyz = torch.einsum("ij,bjhw->bihw", m, lin)
    w = torch.tensor(_WHITE, dtype=lin.dtype, device=lin.device).view(1, 3, 1, 1)
    t = (xyz / w).clamp_min(1e-8)
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = torch.where(t > eps, t ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


@torch.no_grad()
def phototype_signature(x: Tensor) -> Tensor:
    """Four fixed statistics per image, in the order (ITA, L*, b*, a*).

    ITA first because it is the one dermatology uses. The medians are taken over pixels
    rather than the means: a clinical photograph contains background and lesion as well as
    skin, and a median is the cheapest way to report the dominant surface rather than the
    average of surface and lesion.
    """
    lab = _srgb_to_lab(x)
    flat = lab.flatten(2)
    med = flat.median(dim=2).values                      # (B, 3) = L*, a*, b*
    L, a, b = med[:, 0], med[:, 1], med[:, 2]
    ita = torch.atan2(L - 50.0, b.abs().clamp_min(1e-3)) * (180.0 / torch.pi)
    return torch.stack([ita, L, b, a], dim=1)


@dataclass
class InstrumentReport:
    """An instrument together with what it is worth."""

    model: TinyCNN
    n_classes: int
    chance: float
    accuracy: dict[str, float] = field(default_factory=dict)
    classes: list[str] = field(default_factory=list)
    n_train: int = 0
    steps: int = 0

    def line(self) -> str:
        acc = "  ".join(f"{k} {100*v:5.1f}%" for k, v in self.accuracy.items())
        return (f"{self.n_classes}-way, {self.n_train} training images, {self.steps} steps; "
                f"{acc}  (chance {100*self.chance:.1f}%)")


def _batches(x: Tensor, y: Tensor, batch: int, rng: np.random.Generator):
    perm = rng.permutation(x.shape[0])
    for s in range(0, x.shape[0] - batch + 1, batch):
        sel = torch.from_numpy(perm[s:s + batch]).to(x.device).long()
        yield x.index_select(0, sel), y.index_select(0, sel)


def _to_float(images: np.ndarray, idx: np.ndarray, device) -> Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device)
    return x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def _fit(model: TinyCNN, x: Tensor, y: Tensor, *, epochs: int, batch: int, lr: float,
         rng: np.random.Generator) -> int:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, x.shape[0] // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    model.train()
    done = 0
    for _ in range(epochs):
        for xb, yb in _batches(x, y, batch, rng):
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    model.eval()
    return done


@torch.no_grad()
def _accuracy(model: TinyCNN, x: Tensor, y: Tensor, batch: int = 256) -> float:
    if x.shape[0] == 0:
        return float("nan")
    hits = 0
    for s in range(0, x.shape[0], batch):
        hits += int((model(x[s:s + batch]).argmax(1) == y[s:s + batch]).sum())
    return hits / x.shape[0]


# ---------------------------------------------------------------------------
# The population predictor
# ---------------------------------------------------------------------------

def train_phenotype_instrument(raw: FitzRaw, *, exclude: set[str], test_hashes: set[str],
                               device, hold_out_frac: float = 0.15, epochs: int = 12,
                               batch: int = 128, lr: float = 2e-3, seed: int = 11,
                               verbose: bool = True) -> InstrumentReport:
    """Fitzpatrick I-III against IV-VI, the axis the population shift moves along.

    ``exclude`` is every hash that must not be trained on -- the held-out target query
    streams, which are the real reference sets. ``test_hashes`` is every image of a test
    condition; those are excluded from training too and used as a second, harder test set,
    because that is the distribution the instrument is asked to judge at C4.
    """
    rng = np.random.default_rng(seed)
    pt = raw.phototypes
    usable = np.array([i for i, h in enumerate(raw.hashes)
                       if pt[i] in SOURCE_PHOTOTYPES + TARGET_PHOTOTYPES], dtype=np.int64)
    label = np.array([0 if pt[i] in SOURCE_PHOTOTYPES else 1 for i in usable], dtype=np.int64)

    is_test = np.array([raw.hashes[i] in test_hashes for i in usable])
    is_excl = np.array([raw.hashes[i] in exclude for i in usable])

    pool = usable[~is_test & ~is_excl]
    pool_y = label[~is_test & ~is_excl]
    order = rng.permutation(len(pool))
    n_hold = max(1, int(round(hold_out_frac * len(pool))))
    hold, fit = order[:n_hold], order[n_hold:]

    x_fit = _to_float(raw.images, pool[fit], device)
    y_fit = torch.from_numpy(pool_y[fit]).to(device).long()
    model = TinyCNN(2).to(device)
    steps = _fit(model, x_fit, y_fit, epochs=epochs, batch=batch, lr=lr, rng=rng)

    acc = {"held-out train/val conditions": _accuracy(
        model, _to_float(raw.images, pool[hold], device),
        torch.from_numpy(pool_y[hold]).to(device).long())}
    t_idx, t_y = usable[is_test], label[is_test]
    if len(t_idx):
        acc["test conditions (the transfer it is used for)"] = _accuracy(
            model, _to_float(raw.images, t_idx, device),
            torch.from_numpy(t_y).to(device).long())

    rep = InstrumentReport(model=model, n_classes=2, chance=0.5, accuracy=acc,
                           classes=["I-III", "IV-VI"], n_train=len(fit), steps=steps)
    if verbose:
        print(f"  phenotype instrument: {rep.line()}")
        print(f"    trained on train/val-condition images only; {int(is_test.sum())} "
              f"test-condition and {int(is_excl.sum())} held-out-query images withheld")
    return rep


@torch.no_grad()
def phenotype_target_share(rep: InstrumentReport, x: Tensor) -> float:
    """Share of a generated set the population predictor assigns to Fitzpatrick IV-VI.

    This is the Stage C analogue of reading the high-frequency energy of a generated set:
    a coordinate that carries the population moves it towards 1, one that does not leaves it
    where the unconditioned model sits.
    """
    return float((rep.model(x).argmax(1) == 1).float().mean())


# ---------------------------------------------------------------------------
# The semantic predictor
# ---------------------------------------------------------------------------

def train_condition_instrument(raw: FitzRaw, *, exclude: set[str], min_images: int = 24,
                               device=None, hold_out_frac: float = 0.2, epochs: int = 20,
                               batch: int = 128, lr: float = 2e-3, seed: int = 13,
                               must_include: tuple[str, ...] = (),
                               verbose: bool = True) -> InstrumentReport:
    """N-way condition classifier over every condition with at least ``min_images``.

    ``must_include`` names conditions that have to be in the label set whatever their count,
    which is how the test conditions are guaranteed a row: a predictor that cannot name the
    condition it is being asked about would report zero for every arm equally and say nothing.
    """
    rng = np.random.default_rng(seed)
    by_cond: dict[str, list[int]] = {}
    for i, lab in enumerate(raw.labels):
        if lab and raw.hashes[i] not in exclude:
            by_cond.setdefault(lab, []).append(i)
    keep = sorted(c for c, v in by_cond.items()
                  if len(v) >= min_images or c in must_include)
    missing = [c for c in must_include if c not in keep]
    if missing:
        raise ValueError(f"conditions {missing} have no trainable images left after "
                         "exclusion; the semantic instrument cannot name them")
    row_of = {c: i for i, c in enumerate(keep)}

    fit_idx, fit_y, hold_idx, hold_y = [], [], [], []
    for c in keep:
        ids = np.array(by_cond[c], dtype=np.int64)
        ids = ids[rng.permutation(len(ids))]
        n_hold = max(1, int(round(hold_out_frac * len(ids)))) if len(ids) > 2 else 0
        hold_idx += ids[:n_hold].tolist(); hold_y += [row_of[c]] * n_hold
        fit_idx += ids[n_hold:].tolist(); fit_y += [row_of[c]] * (len(ids) - n_hold)

    x_fit = _to_float(raw.images, np.array(fit_idx), device)
    y_fit = torch.tensor(fit_y, device=device).long()
    model = TinyCNN(len(keep)).to(device)
    steps = _fit(model, x_fit, y_fit, epochs=epochs, batch=batch, lr=lr, rng=rng)

    acc = {"held-out": _accuracy(model, _to_float(raw.images, np.array(hold_idx), device),
                                torch.tensor(hold_y, device=device).long())}
    for c in must_include:
        ids = np.array([i for i, y in zip(hold_idx, hold_y) if y == row_of[c]])
        if len(ids):
            acc[c[:22]] = _accuracy(model, _to_float(raw.images, ids, device),
                                    torch.full((len(ids),), row_of[c], device=device).long())

    rep = InstrumentReport(model=model, n_classes=len(keep), chance=1.0 / len(keep),
                           accuracy=acc, classes=keep, n_train=len(fit_idx), steps=steps)
    if verbose:
        print(f"  condition instrument: {rep.line()}")
        print(f"    {len(exclude)} held-out-query images withheld; conditions with fewer "
              f"than {min_images} images are folded out unless named")
    return rep


@torch.no_grad()
def condition_share(rep: InstrumentReport, x: Tensor, condition: str) -> float:
    """Share of a generated set the semantic predictor assigns to ``condition``."""
    if condition not in rep.classes:
        raise KeyError(f"the instrument has no row for '{condition}'")
    row = rep.classes.index(condition)
    return float((rep.model(x).argmax(1) == row).float().mean())
