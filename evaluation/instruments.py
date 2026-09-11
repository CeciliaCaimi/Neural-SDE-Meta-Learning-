"""Measuring instruments for generated images.

Two kinds, deliberately separated by how much they can be trusted.

**Statistics** need no training and cannot drift: three numbers per image that respond to
the three transformations in the task family. Blur removes high-frequency content; added
noise puts broadband energy back; reduced contrast scales everything toward the image
mean. The triple separates all three, and it is the primary instrument here precisely
because there is nothing in it to go wrong.

**A classifier** is a trained instrument and therefore has a ceiling of its own. It is
used only for semantic class, where no cheap statistic exists, and it reports its own
held-out accuracy alongside every verdict so that a low number can be attributed to the
right place. It is trained across all four domains (clean plus the three corruptions),
because generated samples may sit anywhere on the clean-to-corrupted axis and an
instrument that only knows one end would mistake domain for class.

Neither instrument is ever used to decide anything on its own: every reading is a paired
comparison between coordinates, on identical initial noise.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from domains.corruptions import CORRUPTIONS, apply_corruption

# 3x3 discrete Laplacian, the standard high-frequency probe
_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


@torch.no_grad()
def transform_signature(x: Tensor) -> Tensor:
    """Three statistics per image, on (N, C, H, W) in [-1, 1]. Returns (N, 3).

    0. mean |Laplacian|   -- high-frequency energy. Blur suppresses it, noise raises it.
    1. per-image std      -- dynamic range. Contrast reduction suppresses it.
    2. mean |x - blur(x)| -- residual above a local average, a second view of the same
                             high-frequency axis that is less sensitive to isolated edges.

    Blur and contrast both lower statistic 0, but only contrast lowers statistic 1, so the
    triple distinguishes them. Everything here is a fixed convolution: no training, no
    weights, nothing that can silently change between runs.
    """
    c = x.shape[1]
    lap = _LAP.to(x.device, x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    hf = F.conv2d(x, lap, padding=1, groups=c).abs().flatten(1).mean(1)
    sd = x.flatten(1).std(dim=1)
    box = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    resid = (x - box).abs().flatten(1).mean(1)
    return torch.stack([hf, sd, resid], dim=1)


def normalise(feats: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
    """Put the three statistics on a common scale before any distance is taken."""
    return (feats - ref_mean) / ref_std.clamp_min(1e-8)


# ---------------------------------------------------------------------------
# The semantic instrument
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """Small enough to train in minutes, deep enough to read a 32x32 class above chance.

    It is an instrument, not a contribution: its only job is to give the same verdict on
    generated images that it would give on real ones, and to say how often it is right.
    """

    def __init__(self, n_classes: int, width: int = 64) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
                nn.Conv2d(cout, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            )

        self.body = nn.Sequential(
            block(3, width), block(width, width * 2), block(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(width * 4, n_classes)
        self.feature_dim = width * 4

    def features(self, x: Tensor) -> Tensor:
        return self.body(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


def _fetch(images: Tensor, idx: np.ndarray, corruption: str | None, device) -> Tensor:
    gi = torch.from_numpy(np.ascontiguousarray(idx)).to(device).long()
    x = images.index_select(0, gi).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    if corruption is not None:
        gids = torch.from_numpy(np.ascontiguousarray(idx)).to(device)
        x = apply_corruption(x, corruption, 3, gids)
    return x


def train_class_instrument(raw, split, which: str, device, *, hold_out: int = 50,
                           epochs: int = 8, batch: int = 256, lr: float = 2e-3,
                           seed: int = 7, verbose: bool = True):
    """Train the 20-way classifier on one split's classes and report its own ceiling.

    Every image is shown under a randomly chosen domain -- clean or one of the three
    corruptions -- so the instrument reads class rather than domain. The last `hold_out`
    images of each class are never trained on and give the accuracy quoted beside every
    verdict it later produces.
    """
    fids = split.fine_ids(which)
    fine_to_row = {f: i for i, f in enumerate(fids)}
    coarse_of = {f: split.classes[f].coarse_id for f in fids}
    coarses = sorted({coarse_of[f] for f in fids})
    coarse_to_row = {c: i for i, c in enumerate(coarses)}

    rng = np.random.default_rng(seed)
    images = torch.from_numpy(np.ascontiguousarray(raw.images)).to(device)

    train_idx, train_y, test_idx, test_y = [], [], [], []
    for f in fids:
        ids = raw.indices_of_fine(f).astype(np.int32)
        ids = rng.permutation(ids)
        test_idx.append(ids[:hold_out]); test_y += [fine_to_row[f]] * hold_out
        train_idx.append(ids[hold_out:]); train_y += [fine_to_row[f]] * (len(ids) - hold_out)
    train_idx = np.concatenate(train_idx); test_idx = np.concatenate(test_idx)
    train_y = torch.tensor(train_y, device=device)
    test_y_t = torch.tensor(test_y, device=device)

    model = TinyCNN(len(fids)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(train_idx)
    steps = epochs * max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    domains = (None,) + CORRUPTIONS

    model.train()
    step = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n - batch + 1, batch):
            sel = perm[s:s + batch]
            dom = domains[int(rng.integers(len(domains)))]
            x = _fetch(images, train_idx[sel], dom, device)
            if rng.random() < 0.5:                      # horizontal flip, the one cheap augmentation
                x = torch.flip(x, dims=[3])
            loss = F.cross_entropy(model(x), train_y[torch.from_numpy(sel).to(device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1

    # ---- its own ceiling, per domain, on images it never saw --------------------------
    model.eval()
    acc_fine, acc_coarse = {}, {}
    with torch.no_grad():
        for dom in domains:
            preds = []
            for s in range(0, len(test_idx), batch):
                x = _fetch(images, test_idx[s:s + batch], dom, device)
                preds.append(model(x).argmax(1))
            p = torch.cat(preds)
            name = dom or "clean"
            acc_fine[name] = float((p == test_y_t).float().mean())
            pc = torch.tensor([coarse_to_row[coarse_of[fids[i]]] for i in p.tolist()], device=device)
            tc = torch.tensor([coarse_to_row[coarse_of[fids[i]]] for i in test_y], device=device)
            acc_coarse[name] = float((pc == tc).float().mean())

    if verbose:
        print(f"  semantic instrument: {len(fids)}-way over the {which} classes, "
              f"{steps} steps on {n} images")
        print(f"  held-out accuracy on {hold_out} images per class, by domain:")
        for name in acc_fine:
            print(f"    {name:<16} fine {100*acc_fine[name]:5.1f}%   "
                  f"superclass {100*acc_coarse[name]:5.1f}%      "
                  f"(chance {100/len(fids):.0f}% / {100/len(coarses):.0f}%)")

    return model, dict(fids=fids, fine_to_row=fine_to_row, coarse_of=coarse_of,
                       coarse_to_row=coarse_to_row, acc_fine=acc_fine,
                       acc_coarse=acc_coarse, n_fine=len(fids), n_coarse=len(coarses))
