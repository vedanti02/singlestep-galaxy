"""Binned bispectrum estimator (Scoccimarro shell-field FFT method).

The bispectrum is a higher-order statistic never used in training or model
selection, so it is a genuine held-out check: if the super-resolved field only
matched the power spectrum, its bispectrum would still be wrong.

We use the standard shell-field estimator (Scoccimarro 2000): for a wavenumber
shell ``k_i`` build the band-limited field ``I_i(x) = IFFT(delta_k restricted to
the shell)`` and the unit field ``N_i(x) = IFFT(mask_i)``; then

    B(k1,k2,k3) = [ sum_x I_1 I_2 I_3 ] / [ sum_x N_1 N_2 N_3 ]

where the denominator counts the closed triangles. Absolute normalization
cancels when we report ratios to the HF field, which is all we do.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _kgrid(D: int, box: float) -> np.ndarray:
    kx = np.fft.fftfreq(D, d=box / D) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)


def _shell_fields(delta_k: np.ndarray, K: np.ndarray, edges: np.ndarray):
    """Yield (band-limited field I_i, unit field N_i, mean k) per shell."""
    D = delta_k.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (K >= lo) & (K < hi)
        if not mask.any():
            yield None, None, 0.5 * (lo + hi)
            continue
        I = np.fft.ifftn(delta_k * mask).real
        N = np.fft.ifftn(mask.astype(np.float64)).real
        yield I, N, float(K[mask].mean())


def equilateral_bispectrum(delta: np.ndarray, box_size: Optional[float] = None,
                           n_bins: int = 12
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Equilateral bispectrum ``B(k,k,k)`` in ``n_bins`` shells.

    Args:
        delta: ``(D, D, D)`` real overdensity field.
        box_size: physical box length (only sets the k axis).
        n_bins: number of ``|k|`` shells.

    Returns:
        ``(k, B_eq)`` arrays over the shells that contain triangles.
    """
    D = delta.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    K = _kgrid(D, box)
    edges = np.linspace(0.0, K.max(), n_bins + 1)
    dk = np.fft.fftn(delta)
    ks, Bs = [], []
    for I, N, kc in _shell_fields(dk, K, edges):
        if I is None:
            continue
        num = np.sum(I ** 3)
        den = np.sum(N ** 3)
        if den <= 0:
            continue
        ks.append(kc)
        Bs.append(num / den)
    return np.asarray(ks), np.asarray(Bs)


def squeezed_bispectrum(delta: np.ndarray, box_size: Optional[float] = None,
                        n_bins: int = 12, soft_bin: int = 1
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Squeezed bispectrum ``B(k_soft, k, k)`` with one fixed large scale mode.

    ``k_soft`` is fixed to the ``soft_bin``-th shell (a large scale), and the
    other two legs sweep the shells. Uses the same shell-field estimator.
    """
    D = delta.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    K = _kgrid(D, box)
    edges = np.linspace(0.0, K.max(), n_bins + 1)
    dk = np.fft.fftn(delta)
    shells = list(_shell_fields(dk, K, edges))
    Is = shells[soft_bin][0]
    Ns = shells[soft_bin][1]
    if Is is None:
        return np.asarray([]), np.asarray([])
    ks, Bs = [], []
    for I, N, kc in shells:
        if I is None:
            continue
        num = np.sum(Is * I * I)
        den = np.sum(Ns * N * N)
        if den <= 0:
            continue
        ks.append(kc)
        Bs.append(num / den)
    return np.asarray(ks), np.asarray(Bs)
