"""Spatial geometry operations for the Point-Voxel pipeline.

This module is intentionally pure: every function takes plain
``numpy.ndarray`` or ``torch.Tensor`` inputs and produces the same. It
contains the building blocks shared by the dataloader (overlap cropping,
edge masks) and the model (point↔voxel meshing, periodic boundary
conditions on the 3-torus).

Conventions
-----------
* Voxel grids are ordered ``(C, D, D, D)`` (single sample) or
  ``(B, C, D, D, D)`` (batch).
* Point clouds are ordered ``(N, 3)`` (single) or ``(B, N, 3)`` (batch).
* Coordinates passed to point↔voxel functions are normalized to
  ``[0, 1]`` along every axis (so the same code works for any voxel
  resolution).
* The Lagrangian box is treated as a 3-torus with side length
  ``box_size`` (in physical units) — i.e. positions wrap modulo
  ``box_size``. In voxel units the wrap is modulo ``D``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]
Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Edge buffer / overlap utilities
# ---------------------------------------------------------------------------

def inner_crop(grid: ArrayLike, buffer: int) -> ArrayLike:
    """Strip a ``buffer``-voxel band from every face of a 3-D grid.

    The pipeline uses overlapping crops of side ``D`` with overlap ``d``;
    the outer ``d/2`` voxels of each crop are excluded from the loss to
    eliminate stitching artefacts. This helper performs that strip.

    Args:
        grid: Tensor or array of shape ``(..., D, D, D)``. Leading dims
            are preserved (e.g. ``(C, D, D, D)`` or ``(B, C, D, D, D)``).
        buffer: Number of voxels to remove from every face.

    Returns:
        The inner cube, shape ``(..., D - 2*buffer, D - 2*buffer, D - 2*buffer)``.

    Raises:
        ValueError: if ``buffer`` is negative or larger than ``D // 2``.
    """
    D = grid.shape[-1]
    if buffer < 0 or 2 * buffer >= D:
        raise ValueError(f"buffer={buffer} invalid for grid side D={D}")
    if buffer == 0:
        return grid
    s = slice(buffer, D - buffer)
    return grid[..., s, s, s]


def edge_buffer_mask(D: int, buffer: int,
                     dtype: torch.dtype = torch.float32,
                     device: Optional[torch.device] = None) -> Tensor:
    """Build a ``(D, D, D)`` mask that is 1 in the inner cube, 0 in the buffer.

    Used as the per-voxel loss weight when computing the masked MSE — the
    outer ``buffer`` voxels on every face are excluded so that artefacts
    near the crop boundary do not contaminate the gradient.

    Args:
        D: Side length of the crop in voxels.
        buffer: Width of the excluded band on every face (typically
            ``overlap // 2``).
        dtype: Output dtype.
        device: Output device. ``None`` → CPU.

    Returns:
        Tensor of shape ``(D, D, D)`` with ``buffer``-wide zeros at every
        face and ones inside.
    """
    mask = torch.zeros(D, D, D, dtype=dtype, device=device)
    if buffer > 0 and 2 * buffer < D:
        mask[buffer:D - buffer, buffer:D - buffer, buffer:D - buffer] = 1.0
    elif buffer == 0:
        mask.fill_(1.0)
    return mask


def point_in_inner_mask(coords: Tensor, buffer_frac: float) -> Tensor:
    """Boolean mask: 1 if a point lies in the inner sub-cube, 0 otherwise.

    Args:
        coords: ``(B, N, 3)`` or ``(N, 3)`` point coordinates in ``[0, 1]``.
        buffer_frac: Buffer width as a fraction of the side length — i.e.
            ``buffer / D`` if you converted from voxel buffer width.

    Returns:
        Float tensor of the same leading shape (``(B, N)`` or ``(N,)``)
        with values in ``{0.0, 1.0}``.
    """
    if buffer_frac < 0 or buffer_frac >= 0.5:
        raise ValueError(f"buffer_frac must be in [0, 0.5), got {buffer_frac}")
    lo, hi = buffer_frac, 1.0 - buffer_frac
    inner = (coords >= lo) & (coords < hi)
    return inner.all(dim=-1).to(coords.dtype)


def outside_mask_for_crop(env_resolution: int,
                          sim_extent_vox: tuple[int, int, int],
                          crop_origin_vox: tuple[int, int, int],
                          crop_side_vox: int,
                          dtype: np.dtype = np.float32) -> np.ndarray:
    """Indicator mask over the env grid: 1 outside the crop, 0 inside.

    The stitched env cube has resolution ``env_resolution`` per axis and
    summarises a simulation of full-resolution extent ``sim_extent_vox``.
    A crop occupies a sub-cube ``[origin, origin + crop_side)`` in
    full-resolution voxels. This function maps that sub-cube into env
    coordinates (integer division) and returns a binary mask of shape
    ``(env_resolution,) * 3`` that is 0 in the env voxels overlapped by
    the crop and 1 elsewhere — i.e., it identifies the *outside* points.

    Args:
        env_resolution: Side of the env grid (typically 64).
        sim_extent_vox: Full-resolution extent of the simulation
            (e.g. ``(128, 128, 128)`` for a 2×2×2-tile sim).
        crop_origin_vox: Lower-left corner of the crop in full-res voxels.
        crop_side_vox: Crop side length in full-res voxels.
        dtype: Output numpy dtype.

    Returns:
        ``(R, R, R)`` array with values in ``{0, 1}``.
    """
    R = env_resolution
    Lx, Ly, Lz = sim_extent_vox
    sx, sy, sz = crop_origin_vox
    rx, ry, rz = R / Lx, R / Ly, R / Lz
    ex0 = int(np.floor(sx * rx))
    ey0 = int(np.floor(sy * ry))
    ez0 = int(np.floor(sz * rz))
    ex1 = min(R, int(np.ceil((sx + crop_side_vox) * rx)))
    ey1 = min(R, int(np.ceil((sy + crop_side_vox) * ry)))
    ez1 = min(R, int(np.ceil((sz + crop_side_vox) * rz)))
    mask = np.ones((R, R, R), dtype=dtype)
    mask[ex0:ex1, ey0:ey1, ez0:ez1] = 0.0
    return mask


def overlap_crop_starts(L: int, D: int, overlap: int) -> list[int]:
    """Compute crop start indices that cover ``[0, L)`` with the given overlap.

    The last crop is anchored to ``L - D`` if a uniform stride would miss
    the end of the volume.

    Args:
        L: Length of the source volume along an axis (voxels).
        D: Crop side length (voxels).
        overlap: Number of voxels shared between consecutive crops.

    Returns:
        List of start indices.

    Raises:
        ValueError: if ``overlap >= D`` or ``D <= 0``.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if overlap < 0 or overlap >= D:
        raise ValueError(f"overlap must satisfy 0 <= overlap < D ({D}), "
                         f"got {overlap}")
    if L <= D:
        return [0]
    stride = D - overlap
    starts = list(range(0, L - D + 1, stride))
    if starts[-1] + D < L:
        starts.append(L - D)
    return starts


# ---------------------------------------------------------------------------
# Periodic boundary conditions (3-torus)
# ---------------------------------------------------------------------------

def apply_periodic_bc(positions: ArrayLike,
                      box_size: float | Sequence[float]) -> ArrayLike:
    """Wrap positions to the canonical ``[0, box_size)`` interval.

    The cosmological simulation domain is a 3-torus: a particle at
    ``box_size + ε`` is the same particle as one at ``ε``. Use this when
    you have just added a displacement to a Lagrangian position and want
    to fold the result back into the fundamental domain.

    Args:
        positions: Array/tensor of shape ``(..., 3)`` in physical or
            voxel units.
        box_size: Side length of the torus. Either a scalar (cubic box)
            or a length-3 sequence for each axis.

    Returns:
        Same type / shape as ``positions``, every coordinate in
        ``[0, box_size)``.
    """
    if isinstance(positions, torch.Tensor):
        if isinstance(box_size, (int, float)):
            return positions % float(box_size)
        bs = torch.as_tensor(box_size, dtype=positions.dtype,
                             device=positions.device)
        return positions % bs
    if isinstance(box_size, (int, float)):
        return np.mod(positions, float(box_size))
    return np.mod(positions, np.asarray(box_size, dtype=positions.dtype))


def minimum_image(delta: ArrayLike,
                  box_size: float | Sequence[float]) -> ArrayLike:
    """Map separations to the minimum-image convention on the 3-torus.

    For every component, returns ``delta - box * round(delta / box)`` so
    the absolute value is at most ``box / 2``. Useful when computing
    pairwise distances or two-point statistics across the periodic
    boundary.

    Args:
        delta: ``(..., 3)`` separation vectors.
        box_size: Scalar or length-3 sequence.

    Returns:
        Same shape/type with components in ``(-box/2, +box/2]``.
    """
    if isinstance(delta, torch.Tensor):
        bs = (torch.as_tensor(box_size, dtype=delta.dtype, device=delta.device)
              if not isinstance(box_size, (int, float))
              else float(box_size))
        return delta - bs * torch.round(delta / bs)
    bs = float(box_size) if isinstance(box_size, (int, float)) else \
        np.asarray(box_size, dtype=delta.dtype)
    return delta - bs * np.round(delta / bs)


# ---------------------------------------------------------------------------
# Point-to-voxel meshing
# ---------------------------------------------------------------------------

def sample_lagrangian_points(D: int, n_points: int,
                             rng: Optional[np.random.Generator] = None,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample ``n_points`` Lagrangian-grid cells inside a crop.

    Args:
        D: Side length of the crop in voxels.
        n_points: Number of cells to sample (with replacement).
        rng: Optional numpy RNG for reproducibility.

    Returns:
        Tuple ``(idx, coords)`` where:

        * ``idx`` — ``(n_points, 3)`` int64 array of voxel indices in
          ``[0, D)``.
        * ``coords`` — ``(n_points, 3)`` float32 array of normalized
          coordinates in ``[0, 1]`` (cell centres).
    """
    rng = rng or np.random.default_rng()
    idx = rng.integers(0, D, size=(n_points, 3)).astype(np.int64)
    coords = ((idx.astype(np.float32) + 0.5) / D).astype(np.float32)
    return idx, coords


def voxel_to_points_indices(grid: Tensor, idx: Tensor) -> Tensor:
    """Gather per-point features from a voxel grid given integer indices.

    Args:
        grid: ``(B, C, D, D, D)`` voxel features.
        idx: ``(B, N, 3)`` int64 indices in ``[0, D)``.

    Returns:
        ``(B, C, N)`` per-point features.
    """
    B, C, D, _, _ = grid.shape
    N = idx.shape[1]
    b = torch.arange(B, device=grid.device).view(B, 1).expand(B, N)
    return grid[b, :, idx[..., 0], idx[..., 1], idx[..., 2]].transpose(1, 2)


def trilinear_devoxelize(grid: Tensor, coords: Tensor) -> Tensor:
    """Differentiable trilinear gather: voxel grid → per-point features.

    The 8 corner contributions are computed explicitly; this matches
    ``torch.nn.functional.grid_sample`` with ``mode='bilinear'``,
    ``align_corners=False`` and ``padding_mode='border'`` but uses our
    ``(x, y, z)`` axis convention (``coords[..., 0]`` indexes the first
    spatial axis of ``grid``).

    Args:
        grid: ``(B, C, R, R, R)`` voxel grid.
        coords: ``(B, N, 3)`` normalized point coordinates in ``[0, 1]``.

    Returns:
        ``(B, C, N)`` interpolated features.
    """
    B, C, R = grid.shape[0], grid.shape[1], grid.shape[2]
    N = coords.shape[1]
    pos = (coords * R - 0.5).clamp(min=0, max=R - 1)
    p0 = pos.floor().long()
    p1 = (p0 + 1).clamp(max=R - 1)
    f = pos - p0.float()
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    b_idx = torch.arange(B, device=grid.device).view(B, 1).expand(B, N)

    out = grid.new_zeros(B, N, C)
    for dx in (0, 1):
        wx = fx if dx else (1.0 - fx)
        ix = p1[..., 0] if dx else p0[..., 0]
        for dy in (0, 1):
            wy = fy if dy else (1.0 - fy)
            iy = p1[..., 1] if dy else p0[..., 1]
            for dz in (0, 1):
                wz = fz if dz else (1.0 - fz)
                iz = p1[..., 2] if dz else p0[..., 2]
                w = (wx * wy * wz).unsqueeze(-1)                # (B, N, 1)
                vals = grid[b_idx, :, ix, iy, iz]               # (B, N, C)
                out = out + vals * w
    return out.transpose(1, 2).contiguous()


def points_to_voxel(coords: Tensor, features: Tensor, R: int,
                    reduction: str = "mean") -> Tensor:
    """Scatter per-point features back onto a voxel grid.

    Args:
        coords: ``(B, N, 3)`` point coordinates in ``[0, 1]``.
        features: ``(B, N, C)`` per-point features.
        R: Output voxel grid resolution.
        reduction: ``"mean"`` (average per cell) or ``"sum"`` (additive
            deposit, like CIC). ``"mean"`` is the right choice for the
            consistency loss; ``"sum"`` for density deposits.

    Returns:
        ``(B, C, R, R, R)`` voxel grid.

    Raises:
        ValueError: if ``reduction`` is not one of ``{"mean", "sum"}``.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    B, N, C = features.shape
    idx = (coords * R).long().clamp(0, R - 1)                     # (B, N, 3)
    flat = idx[..., 0] * R * R + idx[..., 1] * R + idx[..., 2]    # (B, N)
    out = features.new_zeros(B, C, R * R * R)
    out.scatter_add_(2, flat.unsqueeze(1).expand(-1, C, -1),
                     features.transpose(1, 2))
    if reduction == "mean":
        cnt = features.new_zeros(B, 1, R * R * R)
        ones = features.new_ones(B, 1, N)
        cnt.scatter_add_(2, flat.unsqueeze(1), ones)
        out = out / cnt.clamp(min=1.0)
    return out.view(B, C, R, R, R)
