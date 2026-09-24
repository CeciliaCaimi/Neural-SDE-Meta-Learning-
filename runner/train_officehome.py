"""Office-Home Product->Clipart training entry (note section 3 headline config):
set encoder + FiLM conditioning + linear source->target transport, no refinement, 64x64.

    OFFICEHOME_ROOT=/path python -m runner.train_officehome --run-name oh64 --steps 50000 --ckpt-every 5000

Reuses runner.train's parser/config and the shared training.loop.train via the model_cls seam.
"""
from __future__ import annotations
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import models.film_unet  # noqa: E402,F401  register film_unet
from baselines.film_conditioning import FiLMScoreModel   # noqa: E402
from runner.train import configure, parse                # noqa: E402
from training.loop import train                          # noqa: E402


def main() -> None:
    a = parse()
    cfg = configure(a)
    k = cfg.model.k if a.k is not None else 16
    cfg.model.k = k
    cfg.episodes.scheme = "officehome"
    cfg.episodes.officehome_root = os.environ.get("OFFICEHOME_ROOT", cfg.episodes.officehome_root)
    cfg.model.backbone = "film_unet"
    cfg.model.transport_kind = "linear"
    S = int(os.environ.get("OH_IMAGE_SIZE", "64"))
    cfg.model.backbone_kwargs = dict(
        image_size=S, base_channels=64, channel_mult=(1, 2, 2, 2),
        num_res_blocks=2, attn_resolutions=(16,), k=k,
    )
    # headline defaults (note section 3); CLI can still override steps/ckpt-every via configure
    cfg.train.lr = 2e-4
    cfg.train.ema_decay = 0.999
    if a.steps is None:
        cfg.train.steps = 50000
    print(f"officehome root={cfg.episodes.officehome_root} k={k} transport=linear backbone=film_unet 64x64")
    train(cfg, model_cls=FiLMScoreModel)


if __name__ == "__main__":
    main()
