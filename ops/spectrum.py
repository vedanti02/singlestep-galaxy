"""Isotropic 3D power-spectrum utilities (P(k), T(k), coherence).

All functions take real scalar fields ``(D, D, D)`` and return
``(k_centers, P)`` arrays binned to ``n_bins`` shells in :math:`|k|`.

The default ``box_size = D`` returns wavenumbers in inverse voxel
units; pass the physical box length (e.g. 1000 Mpc/h for Quijote) to
get :math:`k` in physical units.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _kgrid(D: int, box: float) -> np.ndarray:
    kx = np.fft.fftfreq(D, d=box / D) * 2 * np.pi
    ky = np.fft.fftfreq(D, d=box / D) * 2 * np.pi
    kz = np.fft.rfftfreq(D, d=box / D) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)


def _bin_radial(p3d: np.ndarray, K: np.ndarray, n_bins: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, K.max(), n_bins + 1)
    digit = np.clip(np.digitize(K.ravel(), edges) - 1, 0, n_bins - 1)
    counts = np.bincount(digit, minlength=n_bins).astype(float)
    P = np.bincount(digit, weights=p3d.ravel(),
                    minlength=n_bins) / np.maximum(counts, 1)
    k_c = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    return k_c[valid], P[valid]


def power_spectrum(field: np.ndarray, box_size: Optional[float] = None,
                   n_bins: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic auto-power spectrum of a real scalar field."""
    D = field.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    n_bins = n_bins or D // 2
    fk = np.fft.rfftn(field) / D ** 3
    p3d = (fk * fk.conj()).real * (box ** 3)
    return _bin_radial(p3d, _kgrid(D, box), n_bins)


def cross_power(f1: np.ndarray, f2: np.ndarray,
                box_size: Optional[float] = None,
                n_bins: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic cross-power spectrum of two real fields."""
    D = f1.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    n_bins = n_bins or D // 2
    fk1 = np.fft.rfftn(f1) / D ** 3
    fk2 = np.fft.rfftn(f2) / D ** 3
    p3d = (fk1 * fk2.conj()).real * (box ** 3)
    return _bin_radial(p3d, _kgrid(D, box), n_bins)


def transfer_function(f_pred: np.ndarray, f_true: np.ndarray,
                      box_size: Optional[float] = None,
                      n_bins: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(k, T(k), P_pred(k), P_true(k))`` where ``T = sqrt(P_pred / P_true)``."""
    k, Pp = power_spectrum(f_pred, box_size, n_bins)
    _, Pt = power_spectrum(f_true, box_size, n_bins)
    T = np.sqrt(np.clip(Pp / np.clip(Pt, 1e-30, None), 0, None))
    return k, T, Pp, Pt


def coherence(f_pred: np.ndarray, f_true: np.ndarray,
              box_size: Optional[float] = None,
              n_bins: Optional[int] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-coherence ``r(k) = P_x / sqrt(P_pred * P_true)`` ∈ ``[-1, 1]``."""
    k, Px = cross_power(f_pred, f_true, box_size, n_bins)
    _, P1 = power_spectrum(f_pred, box_size, n_bins)
    _, P2 = power_spectrum(f_true, box_size, n_bins)
    return k, Px / np.sqrt(np.clip(P1 * P2, 1e-30, None))
