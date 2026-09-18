"""Training entry point. The only one -- put one-off scripts in scripts/, not at the root.

    python -m runner.train                       # default configuration
    python -m runner.train --steps 200 --smoke   # smoke test: small model, a few hundred steps
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.base_config import BaseConfig                 # noqa: E402
from training.loop import train                           # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-Diffusion training")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--k", type=int, default=None, help="latent dimension; sweep {2,4,8,16,32}")
    p.add_argument("--backbone", type=str, default=None, help="registry name, e.g. small_unet")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-z", type=float, default=None, help="coordinate regularisation lambda_z")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--center-coords", action="store_true",
                   help="E13: centred coordinates z = z_center + delta_z (absorb the shared offset)")
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--diagnose-every", type=int, default=None)
    p.add_argument("--pooling", choices=("mean", "mean_std"), default=None,
                   help="set pooling; mean_std adds second moments")
    p.add_argument("--base-channels", type=int, default=None,
                   help="backbone base channels; shrinking it tests the capacity-absorption diagnosis")
    p.add_argument("--severity", type=int, choices=(1,2,3,4,5), default=None,
                   help="transformation strength; used by the relatedness sweep")
    p.add_argument("--m-source", type=int, default=None,
                   help="M_S: source images fed to the encoder (source-evidence sweep)")
    p.add_argument("--scheme",
                   choices=("sibling", "domainshift", "fitzpatrick", "chestxray"), default=None,
                   help="split scheme; domainshift = clean to corrupted, fitzpatrick = "
                        "Stage C's skin-phototype shift (one relation), chestxray = Stage "
                        "C's age shift on ChestX-ray14 (three relations)")
    p.add_argument("--cxr-path", type=str, default=None,
                   help="which ChestX-ray14 task split to train on; pass it explicitly, as "
                        "the split file fixes the task partition and the relation set")
    p.add_argument("--fitz-path", type=str, default=None,
                   help="which Fitzpatrick condition split to train on. As with the "
                        "domain-shift file, pass it explicitly: the split file fixes the "
                        "task partition, so two runs that disagree here are not comparable.")
    p.add_argument("--domainshift-path", type=str, default=None,
                   help="which domain-shift split file to train on. Pass it explicitly "
                        "rather than relying on the default: the split file fixes the "
                        "task family, so two runs that disagree here are not comparable.")
    p.add_argument("--transport-kind", choices=("residual_mlp", "identity", "constant", "linear"),
                   default=None,
                   help="A5, the transport-structure ablation. Each variant is trained "
                        "jointly, as the current map was, so that the comparison is "
                        "between structures and not between training objectives.")
    p.add_argument("--film-mode", choices=("per_block", "temb"), default=None,
                   help="A2, read only by runner/train_film.py. per_block gives each "
                        "residual block its own [gamma_j, beta_j] = W_j z; temb folds z "
                        "into the shared timestep embedding, which is what E15 measured "
                        "on the denoising loss.")
    p.add_argument("--smoke", action="store_true", help="small model, few steps; checks the pipeline only")
    return p.parse_args()


def configure(a: argparse.Namespace) -> BaseConfig:
    cfg = BaseConfig()

    if a.smoke:
        cfg.model.backbone_kwargs = dict(
            base_channels=32, channel_mult=(1, 2), num_res_blocks=1, attn_resolutions=()
        )
        cfg.model.k = 8
        cfg.episodes.enc_source_images = 16
        cfg.episodes.query_batch = 8
        cfg.train.steps = 200
        cfg.train.log_every = 20
        cfg.train.diagnose_every = 100
        cfg.train.ckpt_every = 10 ** 9      # no checkpoints during a smoke test
        cfg.train.warmup_steps = 20
        cfg.run_name = "smoke"

    if a.steps is not None:
        cfg.train.steps = a.steps
    if a.k is not None:
        cfg.model.k = a.k
    if a.backbone is not None:
        cfg.model.backbone = a.backbone
    if a.lr is not None:
        cfg.train.lr = a.lr
    if a.lambda_z is not None:
        cfg.train.lambda_z = a.lambda_z
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.device is not None:
        cfg.device = a.device
    if a.no_amp:
        cfg.train.amp = False
    if a.center_coords:
        cfg.model.center_coords = True
    if a.ckpt_every is not None:
        cfg.train.ckpt_every = a.ckpt_every
    if a.diagnose_every is not None:
        cfg.train.diagnose_every = a.diagnose_every
    if a.pooling is not None:
        cfg.model.encoder_pooling = a.pooling
    if a.transport_kind is not None:
        cfg.model.transport_kind = a.transport_kind
    if a.scheme is not None:
        cfg.episodes.scheme = a.scheme
    if a.domainshift_path is not None:
        cfg.episodes.domainshift_path = a.domainshift_path
    if a.fitz_path is not None:
        cfg.episodes.fitz_path = a.fitz_path
    if a.cxr_path is not None:
        cfg.episodes.cxr_path = a.cxr_path
    if a.severity is not None:
        cfg.episodes.severity_override = a.severity
    if a.m_source is not None:
        cfg.episodes.enc_source_images = a.m_source
    if a.base_channels is not None:
        cfg.model.backbone_kwargs = dict(cfg.model.backbone_kwargs)
        cfg.model.backbone_kwargs["base_channels"] = a.base_channels
        if a.base_channels <= 48:          # a small backbone drops one downsample level and attention
            cfg.model.backbone_kwargs["channel_mult"] = (1, 2)
            cfg.model.backbone_kwargs["num_res_blocks"] = 1
            cfg.model.backbone_kwargs["attn_resolutions"] = ()

    # Resolve split file paths relative to the package root
    if not os.path.isabs(cfg.episodes.split_path):
        cfg.episodes.split_path = os.path.join(_ROOT, cfg.episodes.split_path)
    if not os.path.isabs(cfg.episodes.domainshift_path):
        cfg.episodes.domainshift_path = os.path.join(_ROOT, cfg.episodes.domainshift_path)
    if not os.path.isabs(cfg.episodes.fitz_path):
        cfg.episodes.fitz_path = os.path.join(_ROOT, cfg.episodes.fitz_path)
    if not os.path.isabs(cfg.episodes.cxr_path):
        cfg.episodes.cxr_path = os.path.join(_ROOT, cfg.episodes.cxr_path)
    if not os.path.isabs(cfg.train.ckpt_dir):
        cfg.train.ckpt_dir = os.path.join(_ROOT, cfg.train.ckpt_dir)

    return cfg


def main() -> None:
    train(configure(parse()))


if __name__ == "__main__":
    main()
