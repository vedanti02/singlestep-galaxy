"""Cloud-in-cell density deposit from a Lagrangian displacement field."""

from __future__ import annotations

from typing import Optional

import numpy as np

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
