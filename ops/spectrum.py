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


def amplitude_match(f_pred: np.ndarray, f_true: np.ndarray,
                    box_size: Optional[float] = None,
                    n_bins: Optional[int] = None) -> np.ndarray:
    """Rescale ``f_pred``'s Fourier amplitude per k-shell to match ``f_true``'s power.

    ORACLE (uses the true field's power spectrum) — an *upper bound* on what a
    phase-preserving spectral amplitude correction can achieve, not a
    deployable step (a deployable version predicts ``T(k)`` from cosmology).

    Multiplying every mode in a radial shell by the real scalar
    ``a(k) = sqrt(P_true(k) / P_pred(k))`` sets ``P_corrected(k) = P_true(k)``
    (so ``T(k) -> 1``) while leaving the phases untouched. Because the
    cross-coherence ``r(k) = <pred, true*> / sqrt(P_pred P_true)`` is invariant
    under a real per-shell rescaling of ``pred``, ``r(k)`` is *exactly
    preserved*. This isolates the amplitude (power) error from the phase
    (coherence) error.

    Args:
        f_pred: ``(D, D, D)`` predicted real field.
        f_true: ``(D, D, D)`` ground-truth real field.
        box_size: physical box length (only sets the k-grid; result is
            independent of it since binning is monotone in |k|).
        n_bins: number of radial shells for the correction. Use a *fine*
            binning (default ``D // 2``) so amplitude is corrected at all
            scales; the downstream T/r evaluation can use its own coarser bins.

    Returns:
        ``(D, D, D)`` amplitude-corrected real field (same phases as f_pred).
    """
    D = f_pred.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    n_bins = n_bins or D // 2
    fp = np.fft.rfftn(f_pred)
    ft = np.fft.rfftn(f_true)
    K = _kgrid(D, box)
    edges = np.linspace(0.0, K.max(), n_bins + 1)
    digit = np.clip(np.digitize(K.ravel(), edges) - 1, 0, n_bins - 1)
    Pp = np.bincount(digit, weights=(np.abs(fp) ** 2).ravel(), minlength=n_bins)
    Pt = np.bincount(digit, weights=(np.abs(ft) ** 2).ravel(), minlength=n_bins)
    scale = np.sqrt(np.divide(Pt, np.clip(Pp, 1e-30, None)))
    fp_corr = fp * scale[digit].reshape(fp.shape)
    return np.fft.irfftn(fp_corr, s=f_pred.shape)


def rescale_by_curve(field: np.ndarray, k_curve: np.ndarray,
                     T_curve: np.ndarray, box_size: Optional[float] = None
                     ) -> np.ndarray:
    """DEPLOYABLE phase-preserving amplitude correction from a *fixed* T(k) curve.

    Unlike :func:`amplitude_match` (which reads the true field's power, an
    oracle), this uses a pre-measured transfer-function curve ``T_curve(k)`` —
    e.g. the model→HF transfer function averaged over *training* simulations —
    to boost the prediction's amplitude by ``a(k) = 1 / T_curve(|k|)``, per mode
    and interpolated in ``|k|``. It touches only Fourier *magnitude*, so the
    cross-coherence ``r(k)`` is preserved exactly (same invariance argument as
    ``amplitude_match``); the transfer curve carries *no* information from the
    field being corrected, so applying a training-derived curve to a held-out
    validation field is a genuine, non-circular deployable test.

    Args:
        field: ``(D, D, D)`` predicted real field.
        k_curve: ``(M,)`` monotone wavenumbers of the transfer curve.
        T_curve: ``(M,)`` transfer function ``sqrt(P_pred/P_HF)`` at ``k_curve``.
        box_size: physical box length (sets the mode wavenumbers).

    Returns:
        ``(D, D, D)`` amplitude-corrected field (phases unchanged).
    """
    D = field.shape[0]
    box = float(box_size) if box_size is not None else float(D)
    fk = np.fft.rfftn(field)
    K = _kgrid(D, box)
    Tk = np.interp(K, k_curve, T_curve, left=T_curve[0], right=T_curve[-1])
    a = 1.0 / np.clip(Tk, 1e-3, None)
    return np.fft.irfftn(fk * a, s=field.shape)
