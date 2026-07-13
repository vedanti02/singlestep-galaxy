"""Per-channel normalization statistics for displacement fields.

Two kinds of statistics live here:

* **Input stats** (``mean``/``std``): per-channel moments of the LF
  displacement field, computed from the actual full-resolution LF
  *tiles* the model ingests (NOT the block-mean-pooled ``stitched/``
  env cubes — mean pooling suppresses small-scale variance by >3x and
  produced badly mis-scaled stats in early runs).
* **Residual stats** (``res_mean``/``res_std``): per-channel moments of
  the physical HF−LF residual, computed over paired train tiles. The
  training target is standardized with these so it has ~unit variance,
  matching the N(0,1) flow-matching prior. (The old code divided the
  residual by the LF-field std, leaving the target std at ~0.4–0.8 —
  a known driver of under-dispersed single-step predictions.)

Backward compatibility: checkpoints saved before residual stats existed
deserialize with ``res_mean = 0`` and ``res_std = std``, which makes
``normalize_residual``/``denormalize_residual`` reproduce the old
``(hf - lf) / std`` convention bit-for-bit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class NormStats:
    """Per-channel input + residual normalization for a 3-vector field.

    Attributes:
        mean: ``(3,)`` channel means of the LF input field.
        std:  ``(3,)`` channel std-devs of the LF input field.
        res_mean: ``(3,)`` channel means of the HF−LF residual
            (``None`` → zeros; legacy checkpoints).
        res_std:  ``(3,)`` channel std-devs of the HF−LF residual
            (``None`` → falls back to ``std``; legacy checkpoints).
    """
    mean: np.ndarray
    std:  np.ndarray
    res_mean: Optional[np.ndarray] = field(default=None)
    res_std:  Optional[np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        if self.res_mean is None:
            self.res_mean = np.zeros_like(np.asarray(self.mean))
        if self.res_std is None:
            self.res_std = np.asarray(self.std).copy()

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Apply ``(arr - mean) / std`` with broadcast over (3, D, D, D)."""
        m = self.mean.reshape(3, 1, 1, 1)
        s = self.std.reshape(3, 1, 1, 1)
        return (arr - m) / s

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        m = self.mean.reshape(3, 1, 1, 1)
        s = self.std.reshape(3, 1, 1, 1)
        return arr * s + m

    def normalize_residual(self, arr: np.ndarray) -> np.ndarray:
        """Standardize a physical HF−LF residual field ``(3, ...)``."""
        m = self.res_mean.reshape(3, *([1] * (arr.ndim - 1)))
        s = self.res_std.reshape(3, *([1] * (arr.ndim - 1)))
        return (arr - m) / s

    def denormalize_residual(self, arr: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize_residual` for ``(3, ...)`` arrays."""
        m = self.res_mean.reshape(3, *([1] * (arr.ndim - 1)))
        s = self.res_std.reshape(3, *([1] * (arr.ndim - 1)))
        return arr * s + m

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "res_mean": self.res_mean.tolist(),
                "res_std": self.res_std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        def _opt(key):
            v = d.get(key)
            return None if v is None else np.asarray(v, dtype=np.float32)
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"],  dtype=np.float32),
                   res_mean=_opt("res_mean"),
                   res_std=_opt("res_std"))


def _channel_moments(paths: Iterable[str], max_files: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate per-channel (mean, E[x^2]) over up to ``max_files`` arrays."""
    means = np.zeros(3, dtype=np.float64)
    sqs   = np.zeros(3, dtype=np.float64)
    n = 0
    for p in list(paths)[:max_files]:
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float64)
        means += a.mean(axis=(1, 2, 3))
        sqs   += (a ** 2).mean(axis=(1, 2, 3))
        n += 1
    return means, sqs, n


def compute_norm_stats(lf_paths: Iterable[str],
                       max_files: int = 16) -> NormStats:
    """Per-channel input stats from a sample of LF displacement arrays.

    Pass full-resolution LF *tile* paths (the fields ``__getitem__``
    actually normalizes). Callers should sample tiles across many sets
    (e.g. one tile per set) so the estimate spans the cosmology range.

    Args:
        lf_paths: Iterable of absolute paths to ``(3, D, D, D)`` ``.npy``
            arrays.
        max_files: Cap on how many files to read.

    Returns:
        :class:`NormStats` with float32 arrays (residual stats left at
        their legacy fallback — use :func:`compute_residual_stats` to
        fill them).

    Raises:
        FileNotFoundError: if none of ``lf_paths`` exist — silently
            proceeding used to yield std=1e-4 and ~1e4-scale inputs.
    """
    means, sqs, n = _channel_moments(lf_paths, max_files)
    if n == 0:
        raise FileNotFoundError(
            "compute_norm_stats: none of the provided paths exist; "
            "check data.root (does it contain the LF tile tree?)"
        )
    means /= n
    sqs   /= n
    var = np.clip(sqs - means ** 2, 1e-8, None)
    return NormStats(mean=means.astype(np.float32),
                     std=np.sqrt(var).astype(np.float32))


def compute_residual_stats(pair_paths: Iterable[tuple[str, str]],
                           max_files: int = 16
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel moments of the physical HF−LF residual.

    Args:
        pair_paths: Iterable of ``(lf_path, hf_path)`` tile pairs (same
            set, same tile index).
        max_files: Cap on how many pairs to read.

    Returns:
        ``(res_mean, res_std)`` float32 arrays of shape ``(3,)``.

    Raises:
        FileNotFoundError: if no valid pair exists.
    """
    means = np.zeros(3, dtype=np.float64)
    sqs   = np.zeros(3, dtype=np.float64)
    n = 0
    for lf_p, hf_p in list(pair_paths)[:max_files]:
        if not (os.path.exists(lf_p) and os.path.exists(hf_p)):
            continue
        r = np.load(hf_p).astype(np.float64) - np.load(lf_p).astype(np.float64)
        means += r.mean(axis=(1, 2, 3))
        sqs   += (r ** 2).mean(axis=(1, 2, 3))
        n += 1
    if n == 0:
        raise FileNotFoundError(
            "compute_residual_stats: no valid (lf, hf) tile pair found; "
            "check data.root layout"
        )
    means /= n
    sqs   /= n
    var = np.clip(sqs - means ** 2, 1e-8, None)
    return means.astype(np.float32), np.sqrt(var).astype(np.float32)
