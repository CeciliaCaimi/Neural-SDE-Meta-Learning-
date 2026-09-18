r"""NIH ChestX-ray14: thoracic finding as the semantic task, age group as the population.

Stage C's second attempt, after Fitzpatrick17k answered C4 in the negative for a reason that
was structural rather than incidental: it offers **one** relation, phototype I-III to IV-VI, so
the same shift applies to every episode and a constant coordinate solves the meta-objective.
Age is not binary. One source age bin against several target bins gives several relations,
which is the structure CIFAR has (three corruptions) and where the method works.

`handoff/results/C0_chestxray.txt` is the feasibility gate and fixes the design: source ages
30-45, relations 0-30, 45-60 and 60-75, ten usable tasks split 4/3/3, nine held-out episodes
per K_T.

**The data never enters the repository.** Point at your own copy with CHESTXRAY_ROOT, read at
import exactly as domains/cifar100.py reads CIFAR100_ROOT.

    $env:CHESTXRAY_ROOT = "D:\data\chestxray14"      # PowerShell
    export CHESTXRAY_ROOT=/data/chestxray14           # POSIX

**Images are read out of the archives, not extracted.** The twelve zips are 45 GB and
extracting them would cost ninety, for a split that names a few thousand of the 112 120
images. zipfile reads only the central directory to build the name index, which takes seconds,
and each PNG is decoded on demand and centre-cropped before resizing -- a chest radiograph is
not square and stretching one distorts the cardiothoracic ratio, which is what Cardiomegaly is.

Three contaminations the gate found, all handled before an image is ever read; see
runner/build_cxr_splits.py for where each is enforced:

1. **Multi-label rows are excluded.** "Effusion|Infiltration" belongs to no single task.
2. **Patients that straddle an age bin are dropped** (4.7 %). Follow-ups span up to 14 years,
   and such a patient would put the same chest on both sides of the population shift.
3. **Patients carrying more than one finding are collapsed to their most frequent one**
   (28.2 % of patients). They otherwise leak across the task split, which is the split that
   makes the relation rather than the task the object of generalisation.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RELATIVE_ROOT = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "dataset", "chestxray14")
)
DEFAULT_ROOT = os.environ.get("CHESTXRAY_ROOT") or _RELATIVE_ROOT

#: The fourteen findings. "No Finding" is deliberately not one of them: it is 60 361 of the
#: 91 324 single-label rows and is a different kind of object from a diagnosis.
FINDINGS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
)

IMAGE_SIZE = 32


@dataclass(frozen=True)
class CXRAnnotation:
    filename: str
    finding: str
    patient: str
    age: int
    sex: str
    view: str           # PA or AP -- different geometries; the gate measures the imbalance


def annotation_path(root: str | None = None) -> str:
    return os.path.join(root or DEFAULT_ROOT, "Data_Entry_2017_v2020.csv")


def load_annotations(root: str | None = None) -> dict[str, CXRAnnotation]:
    """The official annotation table, keyed by image filename."""
    path = annotation_path(root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Set CHESTXRAY_ROOT, or run scripts/fetch_cxr.py --only csv.")
    out: dict[str, CXRAnnotation] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                age = int(r["Patient Age"])
            except (TypeError, ValueError):
                age = -1
            out[r["Image Index"]] = CXRAnnotation(
                filename=r["Image Index"], finding=r["Finding Labels"].strip(),
                patient=r["Patient ID"], age=age,
                sex=r["Patient Gender"].strip(), view=r["View Position"].strip())
    return out


# ---------------------------------------------------------------------------
# Reading PNGs out of the archives
# ---------------------------------------------------------------------------

class ZipIndex:
    """filename -> (archive, member), built from the zips' central directories.

    Opening twelve archives and listing them reads only the central directory, not the
    compressed data, so this costs seconds rather than the minutes a full scan would. The
    handles stay open: reopening per image would dominate the decode time.
    """

    #: The release has twelve archives. An index built from fewer is not an error the caller
    #: can be allowed to miss: it would either fail later with a confusing "image is in no
    #: archive", or, worse, let a run train on whatever happened to have arrived.
    N_ARCHIVES = 12

    def __init__(self, root: str | None = None, allow_partial: bool = False) -> None:
        self.root = root or DEFAULT_ROOT
        paths = sorted(
            os.path.join(self.root, f) for f in os.listdir(self.root)
            if f.startswith("images_") and f.endswith(".zip"))
        if not paths:
            raise FileNotFoundError(
                f"no images_*.zip in {self.root}; run scripts/fetch_cxr.py")
        self._zips: dict[str, zipfile.ZipFile] = {}
        self.where: dict[str, tuple[str, str]] = {}
        self.incomplete: list[str] = []
        for p in paths:
            try:
                z = zipfile.ZipFile(p)
            except zipfile.BadZipFile:
                # Almost always a download still in flight: fetch_cxr.py appends, so a
                # partial file has no readable central directory yet.
                self.incomplete.append(os.path.basename(p))
                continue
            self._zips[p] = z
            for member in z.namelist():
                if member.endswith(".png"):
                    self.where[os.path.basename(member)] = (p, member)
        self.archives = list(self._zips)
        missing = self.N_ARCHIVES - len(self.archives)
        if (self.incomplete or missing > 0) and not allow_partial:
            raise RuntimeError(
                f"{len(self.archives)} of {self.N_ARCHIVES} archives are readable in "
                f"{self.root}"
                + (f"; still downloading or corrupt: {', '.join(self.incomplete)}"
                   if self.incomplete else "")
                + ". Let scripts/fetch_cxr.py finish, then retry. Pass allow_partial=True "
                  "only to inspect what has arrived -- never to train or evaluate on it.")

    def __len__(self) -> int:
        return len(self.where)

    def read(self, filename: str) -> bytes:
        if filename not in self.where:
            raise KeyError(
                f"{filename} is in no archive under {self.root}. "
                f"{len(self.where)} images are indexed; the full release has 112120, so an "
                "incomplete download is the usual cause -- re-run scripts/fetch_cxr.py.")
        path, member = self.where[filename]
        return self._zips[path].read(member)

    def close(self) -> None:
        for z in self._zips.values():
            z.close()


@dataclass
class CXRRaw:
    """Decoded pixels plus the annotation of each, addressed by filename."""

    images: np.ndarray                 # (N, S, S, 3) uint8, in the order of `names`
    names: list[str]
    findings: list[str]
    patients: list[str]
    ages: np.ndarray
    views: list[str]
    index_of: dict[str, int]
    image_size: int
    root: str

    def indices(self, names: list[str]) -> np.ndarray:
        missing = [n for n in names if n not in self.index_of]
        if missing:
            raise KeyError(
                f"{len(missing)} images are not loaded, first {missing[:3]}. "
                "load_chestxray(names=...) loads only what it is asked for; pass the union "
                "of every stream you intend to read.")
        return np.array([self.index_of[n] for n in names], dtype=np.int64)


def _decode(blob: bytes, size: int) -> np.ndarray:
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as im:
        im = im.convert("L")            # radiographs are greyscale; some are stored as RGBA
        w, h = im.size
        # Centre-crop to a square before resizing. These are 1024x1024 as released but a
        # few differ; stretching a chest film would distort the cardiothoracic ratio, which
        # is the whole content of one of the fourteen tasks.
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
        a = np.asarray(im, dtype=np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)      # the backbone expects three channels


def load_chestxray(image_size: int = IMAGE_SIZE, names: list[str] | None = None,
                   root: str | None = None, index: ZipIndex | None = None,
                   verbose: bool = True) -> CXRRaw:
    """Decode the named images into memory.

    ``names`` restricts the load, which is what every caller should do: the split reads a few
    thousand of the 112 120. Passing None decodes everything indexed, which is rarely wanted.
    """
    root = root or DEFAULT_ROOT
    ann = load_annotations(root)
    idx = index or ZipIndex(root)
    want = list(names) if names is not None else sorted(idx.where)

    missing = [n for n in want if n not in idx.where]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(want)} requested images are in no archive, first "
            f"{missing[:3]}. Check scripts/fetch_cxr.py finished.")

    if verbose:
        print(f"  chestxray: decoding {len(want)} images at {image_size}x{image_size} "
              f"from {len(idx.archives)} archives in {root}")
    imgs = np.empty((len(want), image_size, image_size, 3), dtype=np.uint8)
    findings, patients, views = [], [], []
    ages = np.empty(len(want), dtype=np.int16)
    for i, n in enumerate(want):
        imgs[i] = _decode(idx.read(n), image_size)
        a = ann.get(n)
        findings.append(a.finding if a else "")
        patients.append(a.patient if a else "")
        views.append(a.view if a else "")
        ages[i] = a.age if a else -1
    return CXRRaw(images=imgs, names=want, findings=findings, patients=patients, ages=ages,
                  views=views, index_of={n: i for i, n in enumerate(want)},
                  image_size=image_size, root=root)
