"""Raw CIFAR-100 loading.

This only turns the on-disk pickles into arrays; no protocol logic lives here.

The official train/test boundary is meaningless for this project: we split **semantic
classes**, not images. All 60000 images are pooled into one array and re-divided by
episodes/splits.py. Each image records its origin (train/test) purely for provenance.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

# Location of dataset/ relative to this file: domains/cifar100.py -> ../../../../dataset
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "dataset", "cifar-100-python")
)

N_FINE = 100
N_COARSE = 20
FINE_PER_COARSE = 5
IMAGES_PER_FINE = 600  # 500 train + 100 test


@dataclass(frozen=True)
class CIFAR100Raw:
    """The pooled dataset. Index 0..59999 is the global image id stored in split files."""

    images: np.ndarray      # (60000, 32, 32, 3) uint8
    fine: np.ndarray        # (60000,) int16  fine label 0..99
    coarse: np.ndarray      # (60000,) int16  coarse label 0..19
    origin: np.ndarray      # (60000,) uint8  0 = official train, 1 = official test
    origin_row: np.ndarray  # (60000,) int32  row index within the original file
    fine_names: tuple[str, ...]
    coarse_names: tuple[str, ...]
    fine_to_coarse: np.ndarray  # (100,) int16

    def indices_of_fine(self, fine_id: int) -> np.ndarray:
        """All global image ids of a fine class, ascending -- independent of read order."""
        return np.flatnonzero(self.fine == fine_id)


def _unpickle(path: str) -> dict:
    with open(path, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def load_cifar100(root: str = DEFAULT_ROOT) -> CIFAR100Raw:
    """Read meta / train / test under cifar-100-python/ and pool them into a CIFAR100Raw."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CIFAR-100 directory not found: {root}")

    meta = _unpickle(os.path.join(root, "meta"))
    fine_names = tuple(n.decode() for n in meta[b"fine_label_names"])
    coarse_names = tuple(n.decode() for n in meta[b"coarse_label_names"])

    parts = []
    for origin_id, split_name in enumerate(("train", "test")):
        d = _unpickle(os.path.join(root, split_name))
        data = d[b"data"]                                  # (N, 3072) uint8, flattened CHW
        n = data.shape[0]
        parts.append(
            dict(
                images=data.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1),
                fine=np.asarray(d[b"fine_labels"], dtype=np.int16),
                coarse=np.asarray(d[b"coarse_labels"], dtype=np.int16),
                origin=np.full(n, origin_id, dtype=np.uint8),
                origin_row=np.arange(n, dtype=np.int32),
            )
        )

    raw = CIFAR100Raw(
        images=np.concatenate([p["images"] for p in parts]),
        fine=np.concatenate([p["fine"] for p in parts]),
        coarse=np.concatenate([p["coarse"] for p in parts]),
        origin=np.concatenate([p["origin"] for p in parts]),
        origin_row=np.concatenate([p["origin_row"] for p in parts]),
        fine_names=fine_names,
        coarse_names=coarse_names,
        fine_to_coarse=_derive_fine_to_coarse(
            np.concatenate([p["fine"] for p in parts]),
            np.concatenate([p["coarse"] for p in parts]),
        ),
    )
    _check_raw(raw)
    return raw


def _derive_fine_to_coarse(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Derive fine->coarse from the data and check it is well defined (one superclass per fine class)."""
    mapping = np.full(N_FINE, -1, dtype=np.int16)
    for f in range(N_FINE):
        cs = np.unique(coarse[fine == f])
        if cs.size != 1:
            raise ValueError(f"fine class {f} maps to several superclasses: {cs.tolist()}")
        mapping[f] = cs[0]
    return mapping


def _check_raw(raw: CIFAR100Raw) -> None:
    """Load-time integrity assertions: corrupt data should fail here, not three hours into training."""
    n = raw.images.shape[0]
    assert n == N_FINE * IMAGES_PER_FINE, f"there should be 60000 images, got {n}"
    assert raw.images.shape[1:] == (32, 32, 3), raw.images.shape
    assert raw.images.dtype == np.uint8
    assert len(raw.fine_names) == N_FINE
    assert len(raw.coarse_names) == N_COARSE

    counts = np.bincount(raw.fine, minlength=N_FINE)
    assert counts.min() == counts.max() == IMAGES_PER_FINE, (
        f"fine class counts are unbalanced: min={counts.min()} max={counts.max()}"
    )

    per_coarse = np.bincount(raw.fine_to_coarse, minlength=N_COARSE)
    assert per_coarse.min() == per_coarse.max() == FINE_PER_COARSE, (
        f"each superclass should hold 5 fine classes, got {per_coarse.tolist()}"
    )

    # Label consistency: each image's coarse label must equal the mapping of its fine label
    assert np.array_equal(raw.coarse, raw.fine_to_coarse[raw.fine]), "coarse labels disagree with the fine->coarse mapping"


def channel_stats(raw: CIFAR100Raw, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalisation constants, for spot-checking the parse: the full set gives mean approx [.5071,.4865,.4409]."""
    img = raw.images if indices is None else raw.images[indices]
    flat = img.reshape(-1, 3).astype(np.float64) / 255.0
    return flat.mean(0), flat.std(0)
