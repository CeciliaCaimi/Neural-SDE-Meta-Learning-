"""A2 training entry point: the generic-latent-conditioning (FiLM) comparison arm.

    python -m runner.train_film --scheme domainshift \
        --domainshift-path artifacts/cifar100_domainshift_c3.json \
        --run-name a2_film --k 16 --steps 50000

It reuses runner.train's argument parser and config assembly verbatim, then changes only
three things: the backbone becomes film_unet, the score model becomes FiLMScoreModel, and
the coordinate dimension is handed to the backbone so it can size its projections.
training/loop.py is **not** touched -- that is what the model_cls seam is for. Centring is
left off for this arm (--center-coords is accepted but the comparison is run without it).

Everything else -- encoder, transport, relation descriptor, episode construction, split,
diffusion schedule, objective, optimiser, budget, seed -- comes from the same code path as
the basis arm, which is the whole point: the arms must differ only in how z reaches the
denoiser.

``score_model`` is written into the config and therefore into every checkpoint, so an
evaluation script rebuilds the FiLM composition rather than the additive basis. Without it
the FiLM state dict loads into the basis model without raising, and every number afterwards
is quietly about the wrong model.
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

    # Swap the backbone to the FiLM U-Net and hand it the coordinate dimension and the
    # insertion point. build_backbone passes model.backbone_kwargs straight into FiLMUNet,
    # so both must live there -- and being in the config they travel in the checkpoint.
    cfg.model.backbone = "film_unet"
    cfg.model.score_model = "film"
    cfg.model.backbone_kwargs = {
        **cfg.model.backbone_kwargs,
        "k": cfg.model.k,
        "film_mode": a.film_mode or "per_block",
    }

    train(cfg, model_cls=FiLMScoreModel)


if __name__ == "__main__":
    main()
