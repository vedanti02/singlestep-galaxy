"""Cloud-in-cell density deposit from a Lagrangian displacement field."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .geometry import apply_periodic_bc


def disp_to_density(disp: np.ndarray, box_size: Optional[float] = None
                    ) -> np.ndarray:
    """Build a cell-centred density contrast :math:`\\delta` from displacement.

    For Lagrangian indices ``q`` on a regular grid of side ``D``,
    final positions are ``x = q * dx + disp`` where ``dx = box / D``,
    wrapped on the 3-torus. Particles are CIC-deposited onto a ``D^3``
    grid and divided by the mean to yield :math:`\\delta = \\rho / \\bar\\rho - 1`.

    Args:
        disp: ``(3, D, D, D)`` displacement field in length units
            consistent with ``box_size``.
        box_size: Side length of the periodic box in the same units as
            ``disp``. ``None`` → use ``D`` (i.e., disp must be in voxel units).

    Returns:
        ``(D, D, D)`` density contrast.
    """
    D = disp.shape[1]
    box = float(box_size) if box_size is not None else float(D)
    dx = box / D
    grid = np.indices((D, D, D), dtype=np.float32)               # (3, D, D, D)
    pos = grid * dx + disp.astype(np.float32)                    # physical units
    pos = apply_periodic_bc(pos, box)
    pos_vox = pos / dx                                           # cells

    rho = np.zeros((D, D, D), dtype=np.float32)
    flat_pos = pos_vox.reshape(3, -1).T                          # (N, 3)
    p0 = np.floor(flat_pos).astype(np.int64) % D
    f  = flat_pos - np.floor(flat_pos)
    p1 = (p0 + 1) % D

    for dx_ in (0, 1):
        for dy_ in (0, 1):
            for dz_ in (0, 1):
                ix = p1[:, 0] if dx_ else p0[:, 0]
                iy = p1[:, 1] if dy_ else p0[:, 1]
                iz = p1[:, 2] if dz_ else p0[:, 2]
                wx = f[:, 0] if dx_ else (1.0 - f[:, 0])
                wy = f[:, 1] if dy_ else (1.0 - f[:, 1])
                wz = f[:, 2] if dz_ else (1.0 - f[:, 2])
                np.add.at(rho, (ix, iy, iz), wx * wy * wz)

    rho_bar = rho.mean()
    return rho / max(float(rho_bar), 1e-12) - 1.0


# Lagrangian index grid memo for cic_density — it is called several times
# per training step, and rebuilding a D^3 meshgrid each call is pure churn.
_GRID_CACHE: dict = {}


def _cached_grid(D: int, device, dtype) -> "torch.Tensor":
    key = (D, str(device), dtype)
    g = _GRID_CACHE.get(key)
    if g is None:
        ar = torch.arange(D, device=device, dtype=dtype)
        g = torch.stack(torch.meshgrid(ar, ar, ar, indexing="ij"))
        _GRID_CACHE[key] = g
    return g


def cic_density(disp_vox: torch.Tensor, periodic: bool = False
                ) -> torch.Tensor:
    """Differentiable, batched CIC deposit from displacement in voxel units.

    Torch twin of :func:`disp_to_density` for use inside training losses /
    discriminators: gradients flow to ``disp_vox`` through the trilinear
    deposit *weights* (the floor indices carry the standard sub-gradient,
    as in any trilinear scatter). Returns raw density ``rho`` normalized
    to unit mean (so ``delta = rho - 1``); the caller picks the transform
    (e.g. ``log(rho + eps)``).

    Boundary handling: training crops are NOT periodic sub-volumes, so by
    default out-of-range deposit corners are clamped to the crop faces.
    The resulting pile-up lives in the outer ~1 voxel band; strip it with
    ``ops.geometry.inner_crop`` before consuming the field. Pass
    ``periodic=True`` only for full-box cubes.

    Args:
        disp_vox: ``(B, 3, D, D, D)`` displacement field in units of
            voxels (physical disp / dx).
        periodic: Wrap deposit indices modulo D instead of clamping.

    Returns:
        ``(B, 1, D, D, D)`` density with mean ~1 per sample.
    """
    if disp_vox.ndim != 5 or disp_vox.shape[1] != 3:
        raise ValueError(f"expected (B, 3, D, D, D), got {tuple(disp_vox.shape)}")
    B, _, D, _, _ = disp_vox.shape
    dev, dtype = disp_vox.device, disp_vox.dtype

    grid = _cached_grid(D, dev, dtype)                                # (3, D, D, D)
    pos = grid.unsqueeze(0) + disp_vox                                # (B, 3, D, D, D)
    if periodic:
        pos = pos % D
    pos = pos.reshape(B, 3, -1).transpose(1, 2)                       # (B, N, 3)

    p0f = pos.floor()
    f = pos - p0f                                                     # in [0, 1)
    p0 = p0f.long()

    rho = disp_vox.new_zeros(B, D * D * D)
    for dx_ in (0, 1):
        ix = p0[..., 0] + dx_
        wx = f[..., 0] if dx_ else (1.0 - f[..., 0])
        for dy_ in (0, 1):
            iy = p0[..., 1] + dy_
            wy = f[..., 1] if dy_ else (1.0 - f[..., 1])
            for dz_ in (0, 1):
                iz = p0[..., 2] + dz_
                wz = f[..., 2] if dz_ else (1.0 - f[..., 2])
                if periodic:
                    jx, jy, jz = ix % D, iy % D, iz % D
                else:
                    jx = ix.clamp(0, D - 1)
                    jy = iy.clamp(0, D - 1)
                    jz = iz.clamp(0, D - 1)
                flat = (jx * D + jy) * D + jz                         # (B, N)
                rho.scatter_add_(1, flat, wx * wy * wz)
    rho = rho.view(B, 1, D, D, D)
    mean = rho.mean(dim=(2, 3, 4), keepdim=True).clamp(min=1e-12)
    return rho / mean
