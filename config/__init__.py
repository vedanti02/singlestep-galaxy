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


class ModelConfig(TypedDict, total=False):
    base_voxel: int               # voxel-encoder width
    base_point: int               # point-trunk width
    cond_dim: int                 # conditioning token dim
    n_blocks: int
    n_style: int                  # cosmology vector length
    env_resolution: int           # stitched LF env grid side
    c_env: int                    # env input channels (3 = disp only,
                                  # 4 = disp + outside-indicator)


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


class FlowConfig(TypedDict, total=False):
    n_steps_train: int            # >1 enables midpoint-sampled t for stability
    n_steps_infer: int            # 1 = single-step
    lambda_voxel: float           # weight of the voxel consistency term


class Config(TypedDict, total=False):
    data:  DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    flow:  FlowConfig


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
