"""Configuration tree: nested dataclasses plus a module-level singleton.

Defaults come from Appendices A.2 / A.3 of the source document. Both the document
(section 15) and this project's protocol require that **every hyperparameter and
adaptation budget be selected on the validation classes and then frozen**; the
test split is reserved for final reporting only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    backbone: str = "small_unet"            # registry name; see models/backbone.py
    backbone_kwargs: dict = field(default_factory=lambda: dict(
        base_channels=128, channel_mult=(1, 2, 2), num_res_blocks=2, attn_resolutions=(16,)
    ))
    k: int = 16                             # A.2 starting value; sweep {2,4,8,16,32}
    basis_init_scale: float = 1e-3

    # Set encoder r_psi
    encoder_width: int = 64
    encoder_feature_dim: int = 256
    encoder_hidden: int = 256
    encoder_pooling: str = "mean"           # mean | mean_std (the latter adds second moments)

    # Transport Delta_gamma -- A.2: (k + d_c) -> 64 -> 64 -> k
    transport_hidden: int = 64
    relation_dim: int = 8                   # d_c in {8, 16}
    # None = a single relation, so c is omitted -- A.2: "Omit c for a single
    # fixed relation".
    #
    # The relation descriptor must be indexed by the *relation* itself (for the
    # domain-shift scheme, the transformation type), never by a semantic class or
    # grouping. A grouping that is disjoint between train and test would make
    # meta-testing read embedding vectors that were never trained, silently
    # disabling the transport network. The document states the rule as
    # "indexed by relation, never semantic class".
    n_relations: int | None = None
    transport_out_moments: bool = False     # the single extension point of A.7; keep False in v1


@dataclass
class DiffusionConfig:
    n_steps: int = 1000
    schedule: str = "cosine"
    loss_weighting: str = "simple"          # simple | snr | min_snr
    min_snr_gamma: float = 5.0


@dataclass
class EpisodeConfig:
    scheme: str = "sibling"                 # sibling | domainshift
    split_path: str = "artifacts/cifar100_split.json"
    domainshift_path: str = "artifacts/cifar100_domainshift.json"
    k_shots: tuple[int, ...] = (1, 2, 5, 10, 20)

    # Number of source images fed to the encoder during training.
    # Note: the streaming pooling described in the document is a *deployment*
    # technique. Training must retain gradients, so M_S is bounded by memory and
    # we subsample from the src_support pool. The full value of a large M_S shows
    # up in the meta-test anchor.
    severity_override: int | None = None     # relatedness / severity sweep; None = use the value stored in the split file
    enc_source_images: int = 64
    query_batch: int = 32                   # query images per step for L_src / L_tgt / L_trans


@dataclass
class TrainConfig:
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup_steps: int = 500

    # L_meta = L_src + lambda_T*L_tgt + lambda_tr*L_trans + lambda_z*L_z   [eq. 30]
    lambda_tgt: float = 1.0                 # A.3: lambda_T = lambda_tr = 1
    lambda_trans: float = 1.0
    lambda_z: float = 1e-4                  # light regularisation against degenerate coordinates

    log_every: int = 50
    diagnose_every: int = 500
    ckpt_every: int = 2_000
    ckpt_dir: str = "checkpoints"
    amp: bool = True                        # bf16 mixed precision


@dataclass
class AdaptConfig:
    """Meta-test refinement budget. **Every strategy shares one object**, so the
    matched comparison the document requires holds structurally rather than by
    manual bookkeeping."""
    steps: int = 25                         # J in {0, 5, 10, 25, 50, 100}
    lr: float = 1e-2                        # eta_z in {1e-3, 3e-3, 1e-2, 3e-2}
    beta0: float = 1.0                      # beta_0 in {0, 0.01, 0.1, 1, 10}
    noise_batch: int = 16                   # (t, eps) draws per step on the support set


@dataclass
class BaseConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    episodes: EpisodeConfig = field(default_factory=EpisodeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)

    device: str = "cuda"
    global_seed: int = 12345
    run_name: str = "meta_diffusion_v1"


cfg = BaseConfig()
