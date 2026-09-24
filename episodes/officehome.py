"""Office-Home Product->Clipart data + episode loader.

One semantic task = one object category. Source domain = Product, target = Clipart, a single
fixed relation (so no relation embedding: n_relations=None). Emits episodes.dataset.EpisodeBatch
so training/meta_train and training/loop are reused unchanged.

Images are decoded from the flwrlabs/office-home parquet shards, resized to image_size, and held
as uint8 (N,3,S,S) per (domain,class); fetched as float [-1,1] like the CIFAR loader.
The frozen within-class partition (partitions.json, content-addressed by sha256) fixes the
source-query / target-query reserves for val/test so a target-query image is never encoded.
"""
from __future__ import annotations
import hashlib, io, json
import numpy as np, torch
from PIL import Image
import pyarrow.parquet as pq

from episodes.dataset import EpisodeBatch


class OfficeHomeData:
    def __init__(self, parquet_paths, split_json, part_json, image_size=64,
                 domains=("Product", "Clipart")):
        sp = json.load(open(split_json))
        self.split = {"train": sp["train"], "val": sp["val"], "test": sp["test"]}
        self.partitions = json.load(open(part_json))
        md = pq.ParquetFile(parquet_paths[0]).schema_arrow.metadata or {}
        hf = next((json.loads(v) for k, v in md.items() if b"huggingface" in k.lower()), {})
        feats = hf.get("info", {}).get("features") or hf.get("features")
        self.class_names = feats["label"]["names"]
        S = image_size
        raw = {(d, c): [] for d in domains for c in self.class_names}
        sha = {(d, c): [] for d in domains for c in self.class_names}
        want = set(domains)
        for p in parquet_paths:
            t = pq.read_table(p, columns=["domain", "label", "image"])
            for d, l, im in zip(t.column("domain").to_pylist(), t.column("label").to_pylist(),
                                t.column("image").to_pylist()):
                if d not in want:
                    continue
                b = im["bytes"]
                a = np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((S, S), Image.BICUBIC))
                raw[(d, self.class_names[l])].append(torch.from_numpy(a.copy()).permute(2, 0, 1))
                sha[(d, self.class_names[l])].append(hashlib.sha256(b).hexdigest())
        self.images = {k: (torch.stack(v).to(torch.uint8) if v else torch.zeros(0, 3, S, S, dtype=torch.uint8))
                       for k, v in raw.items()}
        self.sha = sha
        self.sha_to_idx = {k: {s: i for i, s in enumerate(v)} for k, v in sha.items()}
        self.image_size = S

    def fetch(self, domain, cls, idxs, device):
        u = self.images[(domain, cls)][torch.as_tensor(list(idxs), dtype=torch.long)]
        return u.to(device).float().div_(127.5).sub_(1.0)

    def idx_of_sha(self, domain, cls, shas):
        m = self.sha_to_idx[(domain, cls)]
        return [m[s] for s in shas if s in m]


class OfficeHomeLoader:
    """which='train' draws disjoint support/query from all class images each episode; 'val'/'test'
    use the frozen partition reserves (query never encoded)."""
    def __init__(self, data: OfficeHomeData, which: str, device="cuda", seed=0,
                 src_support=16, src_query=16, tgt_support=10, tgt_query=20):
        self.d = data; self.which = which; self.device = torch.device(device)
        self.ms, self.sq, self.ts, self.tq = src_support, src_query, tgt_support, tgt_query
        self.classes = data.split[which]
        self.rng = np.random.default_rng(seed)

    def _disjoint(self, n_total, a, b):
        perm = self.rng.permutation(n_total)
        return perm[:a], perm[a:a + b]

    def sample(self, k_shot=None):
        cls = self.classes[int(self.rng.integers(len(self.classes)))]
        ts = int(k_shot) if k_shot is not None else self.ts
        S = self.d.image_size
        if self.which == "train":
            nP = self.d.images[("Product", cls)].shape[0]
            nC = self.d.images[("Clipart", cls)].shape[0]
            ss, sq = self._disjoint(nP, self.ms, self.sq)
            tsup, tq = self._disjoint(nC, ts, self.tq)
        else:
            part = self.d.partitions[cls]
            sq = self.d.idx_of_sha("Product", cls, part["src_query"])
            ssup_pool = self.d.idx_of_sha("Product", cls, part["src_support"])
            ss = list(self.rng.choice(ssup_pool, min(self.ms, len(ssup_pool)), replace=False))
            tq = self.d.idx_of_sha("Clipart", cls, part["tgt_query"])
            tsup = (self.d.idx_of_sha("Clipart", cls, part["tgt_support_nested"][str(ts)])
                    if str(ts) in part["tgt_support_nested"]
                    else self.d.idx_of_sha("Clipart", cls, part["tgt_support_pool"])[:ts])
        empty = torch.zeros(0, 3, S, S, device=self.device)
        return EpisodeBatch(
            src_support=self.d.fetch("Product", cls, ss, self.device),
            src_query=self.d.fetch("Product", cls, sq, self.device),
            tgt_support=self.d.fetch("Clipart", cls, tsup, self.device) if len(tsup) else empty,
            tgt_query=self.d.fetch("Clipart", cls, tq, self.device),
            relation=torch.tensor(0, device=self.device, dtype=torch.long),
            provenance={"class": cls, "k": ts},
        )

    def sample_many(self, n, k_shot=None):
        return [self.sample(k_shot) for _ in range(n)]

    def __len__(self):
        return len(self.classes)
