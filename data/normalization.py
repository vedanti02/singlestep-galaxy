"""Per-channel normalization statistics for displacement fields."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class NormStats:
    """Per-channel mean/std for a 3-component displacement field.

    Attributes:
        mean: ``(3,)`` array of channel means.
        std:  ``(3,)`` array of channel std-devs (clipped from below).
    """
    mean: np.ndarray
    std:  np.ndarray

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Apply ``(arr - mean) / std`` with broadcast over (3, D, D, D)."""
        m = self.mean.reshape(3, 1, 1, 1)
        s = self.std.reshape(3, 1, 1, 1)
        return (arr - m) / s

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        m = self.mean.reshape(3, 1, 1, 1)
        s = self.std.reshape(3, 1, 1, 1)
        return arr * s + m

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"],  dtype=np.float32))


def compute_norm_stats(stitched_lf_paths: Iterable[str],
                       max_files: int = 16) -> NormStats:
    """Cheap per-channel stats from a handful of stitched LF cubes.

    The stitched files are tiny (3·64³ floats) and representative of the
    LF distribution, so 8–16 of them give an accurate estimate of the
    per-channel mean and standard deviation.

    Args:
        stitched_lf_paths: Iterable of absolute paths to ``disp.npy``
            files for the LF stitched cubes.
        max_files: Cap on how many files to read.

    Returns:
        :class:`NormStats` with float32 arrays.
    """
    means = np.zeros(3, dtype=np.float64)
    sqs   = np.zeros(3, dtype=np.float64)
    n = 0
    for p in list(stitched_lf_paths)[:max_files]:
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float64)
        means += a.mean(axis=(1, 2, 3))
        sqs   += (a ** 2).mean(axis=(1, 2, 3))
        n += 1
    n = max(n, 1)
    means /= n
    sqs   /= n
    var = np.clip(sqs - means ** 2, 1e-8, None)
    return NormStats(mean=means.astype(np.float32),
                     std=np.sqrt(var).astype(np.float32))
