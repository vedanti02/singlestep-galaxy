"""Strongly-typed configuration objects for the StepOne-PVD library.

Configs are declared as ``TypedDict`` so that downstream code gets full
type-checking under ``mypy`` / ``pyright`` while still loading from a
plain ``yaml`` / ``json`` file. The expected file is
``config/default.yaml`` — call :func:`load_config` to parse and merge
into the typed dict.

Example:
    >>> from config import load_config
    >>> cfg = load_config("config/default.yaml")
    >>> cfg["data"]["crop_size"]
    64
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, TypedDict

try:                              # PyYAML is the standard, but we fall back
    import yaml                   # noqa: F401  -- presence detected at runtime
    _HAVE_YAML = True
except Exception:                 # pragma: no cover
    _HAVE_YAML = False


class DataConfig(TypedDict, total=False):
    root: str                     # e.g. /data/group_data/universedata/lagrangian_output_64
    snapshot: str                 # e.g. PART_009
    crop_size: int                # D — region side; per-crop point count = D**3
    crop_overlap: int             # d — buffer width per face = d/2
    fields: list[str]             # ["disp"] or ["disp", "vel"]
    reader: str                   # "numpy" | "hdf5"
    box_size: float               # length of the periodic box (Mpc/h or voxels)
    env_outside_mask: bool        # if True, mask env to outside-of-crop only and
                                  # append a 4th indicator channel (1=outside)
    augment: bool                 # cube-reflection augmentation (train split)
    norm_sample_tiles: int        # tiles sampled (one per set, strided across
                                  # the train split) for norm stats (default 32)


class ModelConfig(TypedDict, total=False):
    base_voxel: int               # voxel-encoder width
    base_point: int               # point-trunk width
    cond_dim: int                 # conditioning token dim
    n_blocks: int
    n_style: int                  # cosmology vector length
    env_resolution: int           # stitched LF env grid side
    c_env: int                    # env input channels (3 = disp only,
                                  # 4 = disp + outside-indicator)
    c_lf: int                     # LF voxel input channels (3 * n_fields)
    c_lf_pt: int                  # LF point-trunk channels (0 = no skip)
    z_dim: int                    # latent dim for conditional stochasticity
                                  # (0 = deterministic; used by GAN runs)


class OptimConfig(TypedDict, total=False):
    lr: float
    weight_decay: float
    grad_clip: float
    ema_decay: float


class TrainConfig(TypedDict, total=False):
    epochs: int
    batch_size: int
    num_workers: int
    val_every: int
    ckpt_every: int
    seed: int
    device: str
    out_dir: str
    max_train_crops: int          # cap train epoch to N random crops (0 = all)
    max_val_batches: int          # cap validation batches (0 = all)
    spectral_monitor_batches: int # per-epoch crop-level |T-1|/r monitor
                                  # on N val batches (0 = off)
    prefetch_factor: int


class FlowConfig(TypedDict, total=False):
    mode: str                     # "flow_matching" | "lf_init" | "direct"
    n_steps_train: int            # >1 enables midpoint-sampled t for stability
    n_steps_infer: int            # 1 = single-step
    lambda_voxel: float           # weight of the voxel consistency term
    lambda_div: float             # optional divergence (density) regularizer
    t_alpha: float                # Beta(t_alpha, 1) bias toward small t
    p_zero: float                 # fraction of the batch forced to t = 0
    noise_sigma: float            # sqrt(t(1-t))-scaled noise on x_t


class GanConfig(TypedDict, total=False):
    enabled: bool                 # turn on conditional WGAN-GP training
    variant: str                  # "wgan_gp" (hinge_r1 reserved as fallback)
    d_base: int                   # critic base width
    d_lr: float
    d_betas: list[float]
    n_critic: int                 # critic updates per generator update
    gp_lambda: float              # gradient-penalty weight
    lambda_adv_disp: float        # G weight: displacement critic
    lambda_adv_dens: float        # G weight: density critic
    adv_ramp_start_epoch: int     # epochs before the adv term ramps in
    adv_ramp_epochs: int          # ramp length (linear 0 -> lambda)
    density_eps: float            # eps in log(rho + eps)
    d_style_proj: bool            # projection-style cosmology conditioning


class Config(TypedDict, total=False):
    data:  DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    flow:  FlowConfig
    gan:   GanConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

DEFAULT_PATH = Path(__file__).parent / "default.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str | os.PathLike] = None,
                overrides: Optional[dict[str, Any]] = None) -> Config:
    """Load the typed config dict from a YAML file.

    Args:
        path: Path to a YAML file. ``None`` → ``config/default.yaml``.
        overrides: Optional nested dict applied on top of the file contents.
            Useful for CLI overrides (``--data.crop_size 96``).

    Returns:
        A :class:`Config` (TypedDict). Missing keys fall back to whatever
        ``default.yaml`` provides; callers that read keys not present in
        the file should provide their own defaults.

    Raises:
        ImportError: if PyYAML is not available.
        FileNotFoundError: if ``path`` does not exist.
    """
    if not _HAVE_YAML:
        raise ImportError("PyYAML is required to load configs; pip install pyyaml")
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p, "r") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        _deep_update(raw, overrides)
    return raw  # type: ignore[return-value]
