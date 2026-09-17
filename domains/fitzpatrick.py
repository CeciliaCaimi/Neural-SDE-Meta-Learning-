r"""Fitzpatrick17k: skin condition as the semantic task, skin phototype as the population.

Stage C of the final experiment plan. The shift here is not synthetic: the source and target
sides are **different photographs of different patients**, annotated Fitzpatrick I-III and
IV-VI, so there is no clean(X) / transform(X) pair anywhere and no way for the relation to be
partly an identity map. `handoff/STAGE_C_DATA.md` is the account of the data; read it before
writing anything against this module, because three of its facts change the design and not
only its scale.

**The data never enters the repository.** Point at your own copy with FITZPATRICK_ROOT, read
at import exactly as domains/cifar100.py reads CIFAR100_ROOT. The directory holding it on the
machine of record also holds clinical data and a personal identity document, so nothing is
written back into it and nothing above the repository root is added.

    $env:FITZPATRICK_ROOT = "D:\data\fitzpatrick17k"     # PowerShell
    export FITZPATRICK_ROOT=/data/fitzpatrick17k         # POSIX

Three quarters of the published dataset is unreachable: dermaamin.com has lost its DNS
delegation and holds 76 % of the rows. What is on disk is the Atlas Dermatologico share,
3 887 images, of which 3 734 carry a usable phototype. That is why the condition split is
5/3/3 and not the 10/3/3 the annotations would allow.

Images are decoded on demand, centre-cropped to a square and resized. Decoding all 3 887 at
32x32 takes about half a minute, which is nothing beside a training run, so there is no cache
on disk -- a cache would either sit in the repository, where image data must never go, or in
the dataset directory, which is not ours to write to.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RELATIVE_ROOT = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "dataset", "fitzpatrick17k")
)
DEFAULT_ROOT = os.environ.get("FITZPATRICK_ROOT") or _RELATIVE_ROOT

#: The population split. Chosen from the observed counts at C0, not in advance: under
#: I-II against V-VI the reachable archive collapses to three conditions. Any write-up has
#: to say the weaker phenotype contrast is a consequence of data availability.
SOURCE_PHOTOTYPES = (1, 2, 3)
TARGET_PHOTOTYPES = (4, 5, 6)

IMAGE_SIZE = 32


@dataclass(frozen=True)
class FitzAnnotation:
    md5hash: str
    label: str
    phototype: int          # 1..6, or -1 where the annotation is missing

    @property
    def side(self) -> str:
        if self.phototype in SOURCE_PHOTOTYPES:
            return "source"
        if self.phototype in TARGET_PHOTOTYPES:
            return "target"
        return "unknown"


@dataclass
class FitzRaw:
    """Decoded pixels plus the annotation of each, addressed by md5 hash.

    ``images`` is (N, S, S, 3) uint8 in the order of ``hashes``; ``index_of`` maps a hash to
    its row. The split file stores hashes rather than paths or positions, so a hash is the
    only stable identifier and every stream is resolved through ``index_of``.
    """

    images: np.ndarray
    hashes: list[str]
    labels: list[str]
    phototypes: np.ndarray
    index_of: dict[str, int]
    image_size: int
    root: str

    def indices(self, hashes: list[str]) -> np.ndarray:
        missing = [h for h in hashes if h not in self.index_of]
        if missing:
            raise KeyError(
                f"{len(missing)} hashes are not loaded, first {missing[:3]}. "
                "load_fitzpatrick(hashes=...) loads only what it is asked for; pass the "
                "union of every stream you intend to read.")
        return np.array([self.index_of[h] for h in hashes], dtype=np.int64)


def annotation_path(root: str | None = None) -> str:
    return os.path.join(root or DEFAULT_ROOT, "fitzpatrick17k.csv")


def load_annotations(root: str | None = None) -> dict[str, FitzAnnotation]:
    """The published annotation table, keyed by md5 hash. Rows without a phototype keep -1
    rather than being dropped here: whether they are usable is the caller's decision."""
    path = annotation_path(root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Set FITZPATRICK_ROOT to the directory holding "
            "fitzpatrick17k.csv and images/.")
    out: dict[str, FitzAnnotation] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                pt = int(row["fitzpatrick_scale"])
            except (TypeError, ValueError):
                pt = -1
            out[row["md5hash"]] = FitzAnnotation(row["md5hash"], row["label"].strip(), pt)
    return out


def available_hashes(root: str | None = None) -> list[str]:
    """Hashes whose image is actually on disk, sorted so the order is reproducible."""
    d = os.path.join(root or DEFAULT_ROOT, "images")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} not found; run scripts/fetch_fitz.py first")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _decode(path: str, size: int) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # Centre-crop to a square before resizing. These photographs are 460x385 and the
        # like; resizing straight to a square would stretch every lesion horizontally, and
        # the shape of a lesion is part of what the semantic instrument has to read.
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((size, size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def load_fitzpatrick(image_size: int = IMAGE_SIZE, hashes: list[str] | None = None,
                     root: str | None = None, verbose: bool = True) -> FitzRaw:
    """Decode images into memory.

    ``hashes`` restricts the load, which is what a training run should do -- the condition
    split reads about 1 300 of the 3 887 files. Passing None loads everything on disk, which
    is what the frozen instruments want.
    """
    root = root or DEFAULT_ROOT
    ann = load_annotations(root)
    on_disk = set(available_hashes(root))
    want = list(hashes) if hashes is not None else sorted(on_disk)

    missing = [h for h in want if h not in on_disk]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(want)} requested images are not on disk, first "
            f"{missing[:3]}. Three quarters of Fitzpatrick17k is unreachable; "
            "handoff/STAGE_C_DATA.md says which quarter is not.")

    if verbose:
        print(f"  fitzpatrick: decoding {len(want)} images at {image_size}x{image_size} "
              f"from {root}")
    imgs = np.empty((len(want), image_size, image_size, 3), dtype=np.uint8)
    labels, ptypes = [], np.empty(len(want), dtype=np.int16)
    for i, h in enumerate(want):
        imgs[i] = _decode(os.path.join(root, "images", h + ".jpg"), image_size)
        a = ann.get(h)
        labels.append(a.label if a else "")
        ptypes[i] = a.phototype if a else -1
    return FitzRaw(images=imgs, hashes=want, labels=labels, phototypes=ptypes,
                   index_of={h: i for i, h in enumerate(want)},
                   image_size=image_size, root=root)
