"""E15 training entry point: the generic-latent-conditioning (FiLM) comparison arm.

    python -m runner.train_film --scheme domainshift \
        --domainshift-path artifacts/cifar100_domainshift_c3.json \
        --run-name film128 --k 16 --steps 50000

It reuses runner.train's argument parser and config assembly verbatim, then changes only
two things: the backbone becomes film_unet, and z is routed through it by FiLMScoreModel via
the model_cls seam of training.loop.train. training/loop.py is **not** touched -- that is the
seam's purpose. Centring is left off for this arm (pass nothing; --center-coords is accepted
but the comparison is run without it).
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import models.film_unet  # noqa: E402,F401  -- triggers @register_backbone("film_unet")
from baselines.film_conditioning import FiLMScoreModel   # noqa: E402
from runner.train import configure, parse                # noqa: E402
from training.loop import train                          # noqa: E402


def main() -> None:
    a = parse()
    cfg = configure(a)

    # Swap the backbone to the FiLM U-Net and hand it the coordinate dimension. The FiLM
    # network needs k to size z_mlp; build_backbone passes model.backbone_kwargs straight
    # into FiLMUNet, so k must live there.
    cfg.model.backbone = "film_unet"
    cfg.model.backbone_kwargs = {**cfg.model.backbone_kwargs, "k": cfg.model.k}

    train(cfg, model_cls=FiLMScoreModel)


if __name__ == "__main__":
    main()
