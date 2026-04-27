"""SimulationDataset — paired LF/HF crops + global env context.

This module is the only place that knows how the on-disk simulation is
organized. It answers, per index, the same question:

    "Give me one (lf_voxel, hf_residual, env, style, point_sample,
     edge_mask) tuple, ready for the model."

Reading is delegated to a :class:`~data.readers.TileReader`, so the
class is portable across `.npy`/HDF5 layouts.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ops.geometry import (edge_buffer_mask, outside_mask_for_crop,
                          overlap_crop_starts, point_in_inner_mask)

from .normalization import NormStats
from .readers import TileReader, get_reader

SNAPSHOT_DEFAULT = "PART_009"


# ---------------------------------------------------------------------------
# Set discovery — variable per-set extent (1×1×1 up to 6×6×6 tiles)
# ---------------------------------------------------------------------------

def _is_full_box(tiles: list[tuple[int, int, int]]) -> tuple[bool, tuple[int, int, int]]:
    if not tiles:
        return False, (0, 0, 0)
    xs, ys, zs = zip(*tiles)
    extent = (max(xs) + 1, max(ys) + 1, max(zs) + 1)
    full = (len(set(tiles)) == extent[0] * extent[1] * extent[2])
    return full, extent


def discover_sets(root: str,
                  reader: TileReader,
                  snapshot: str = SNAPSHOT_DEFAULT
                  ) -> list[tuple[int, tuple[int, int, int]]]:
    """Return ``[(set_id, extent_xyz_tiles), ...]`` for sets with HF, LF,
    and stitched LF all present and complete.

    Args:
        root: Path that contains ``quijote-64/``, ``quijotelike-64/``, ``stitched/``.
        reader: Tile-format reader.
        snapshot: Snapshot subdirectory (default ``PART_009``).
    """
    hf_idx = reader.index_root(os.path.join(root, "quijote-64"))
    lf_idx = reader.index_root(os.path.join(root, "quijotelike-64"))
    st_root = os.path.join(root, "stitched")
    st_lf = {int(n.replace("set", "").replace("_quijotelike", ""))
             for n in os.listdir(st_root) if n.endswith("_quijotelike")}

    out = []
    for sid in sorted(set(hf_idx) & set(lf_idx) & st_lf):
        hf_full, hf_ext = _is_full_box(hf_idx[sid])
        lf_full, lf_ext = _is_full_box(lf_idx[sid])
        if not (hf_full and lf_full and hf_ext == lf_ext):
            continue
        st_path = os.path.join(st_root, f"set{sid}_quijotelike",
                               snapshot, "disp.npy")
        if not os.path.exists(st_path):
            continue
        out.append((sid, hf_ext))
    return out


def split_sets(sets: list[tuple[int, tuple[int, int, int]]]
               ) -> dict[str, list[tuple[int, tuple[int, int, int]]]]:
    """Last-digit hold-out: 9 → test, 8 → val, else → train."""
    return {
        "train": [s for s in sets if s[0] % 10 not in (8, 9)],
        "val":   [s for s in sets if s[0] % 10 == 8],
        "test":  [s for s in sets if s[0] % 10 == 9],
    }


# ---------------------------------------------------------------------------
# Crop schedule
# ---------------------------------------------------------------------------

def build_crop_index(sets: list[tuple[int, tuple[int, int, int]]],
                     D: int, overlap: int, tile_size: int = 64
                     ) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    """Flat list of crops covering every set with the requested overlap.

    Each entry is ``(set_id, sx, sy, sz, extent_voxels)``.
    """
    flat = []
    for sid, ext in sets:
        Lx, Ly, Lz = ext[0] * tile_size, ext[1] * tile_size, ext[2] * tile_size
        for sx in overlap_crop_starts(Lx, D, overlap):
            for sy in overlap_crop_starts(Ly, D, overlap):
                for sz in overlap_crop_starts(Lz, D, overlap):
                    flat.append((sid, sx, sy, sz, (Lx, Ly, Lz)))
    return flat


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimulationDataset(Dataset):
    """Paired LF/HF Lagrangian-displacement crops + global env context.

    Implements the spec literally:

    1. **Regions of size D × D × D containing all D³ Lagrangian points**
       (no random subsampling). The model budget is therefore D³ — the
       caller chooses ``crop_size`` so that ``D³ ≤`` the model's
       point capacity ``n``. Crops overlap by ``crop_overlap`` voxels in
       every dimension; the outer ``crop_overlap // 2`` voxels of every
       face are excluded from the loss via ``loss_mask`` / ``pt_mask``.
    2. **Outside-only conditioning.** When ``env_outside_mask=True`` the
       env cube is *masked*: env voxels falling inside the current crop
       are set to zero (env values are normalized → mean ≈ 0) and a 4th
       indicator channel is appended (1 where the point lies *outside*
       the crop, 0 inside). The :class:`GlobalContextEncoder` therefore
       sees only the representation of *other* points plus a "where am
       I" hint. Disable to recover the global env baseline.

    Args:
        root: Directory containing ``quijote-64/`` (HF), ``quijotelike-64/``
            (LF), and ``stitched/`` (env) sub-directories.
        sets: Pre-filtered list of ``(set_id, extent_xyz_tiles)``. Use
            :func:`discover_sets` + :func:`split_sets` upstream.
        crop_size: Side length ``D`` of each crop (voxels). ``D³`` must
            be ≤ the model's point capacity.
        crop_overlap: Number of voxels of overlap between adjacent crops.
            Buffer width per face = ``crop_overlap // 2``.
        norm_stats: Per-channel normalization (compute once over training).
        env_outside_mask: If True, mask the env to "outside the crop only"
            and append the indicator channel.
        env_resolution: Side of the env cube (auto-detected if None).
        reader: Storage backend (default :class:`NumpyTileReader`).
        snapshot: Snapshot subdirectory.
        seed: Numpy RNG seed (currently unused — kept for API stability).

    Each ``__getitem__`` returns a dict with tensors:

        ``lf_voxel``  ``(3, D, D, D)``        normalized LF displacement
        ``env``       ``(C_env, R, R, R)``    normalized env (3 channels;
                                              4 if env_outside_mask=True)
        ``coords``    ``(D³, 3)`` in ``[0,1]`` cell-centre coords (every voxel)
        ``lf_pt``     ``(D³, 3)``             LF disp at every cell
        ``tgt_pt``    ``(D³, 3)``             normalized HF–LF residual at every cell
        ``tgt_vox``   ``(3, D, D, D)``        normalized HF–LF residual full grid
        ``loss_mask`` ``(D, D, D)``           1 inside the inner cube
        ``pt_mask``   ``(D³,)``               1 if cell lies in the inner cube
        ``style``     ``(5,)``                cosmology vector
    """

    def __init__(self,
                 root: str,
                 sets: list[tuple[int, tuple[int, int, int]]],
                 crop_size: int,
                 crop_overlap: int,
                 norm_stats: NormStats,
                 env_outside_mask: bool = True,
                 env_resolution: int = 64,
                 reader: Optional[TileReader] = None,
                 snapshot: str = SNAPSHOT_DEFAULT,
                 seed: int = 0) -> None:
        if crop_overlap % 2 != 0 or not (0 < crop_overlap < crop_size):
            raise ValueError(
                f"crop_overlap must be even and in (0, {crop_size}); "
                f"got {crop_overlap}"
            )
        self.root = root
        self.sets = sets
        self.D = crop_size
        self.overlap = crop_overlap
        self.buf = crop_overlap // 2
        self.norm = norm_stats
        self.env_outside_mask = env_outside_mask
        self.env_resolution = env_resolution
        self.reader = reader or get_reader("numpy")
        self.snapshot = snapshot

        self.hf_root = os.path.join(root, "quijote-64")
        self.lf_root = os.path.join(root, "quijotelike-64")
        self.st_root = os.path.join(root, "stitched")

        self.crops = build_crop_index(sets, crop_size, crop_overlap,
                                      tile_size=self.reader.tile_size)

        # pre-compute fixed (D³, 3) cell coords + indices for every crop
        ii, jj, kk = np.meshgrid(np.arange(crop_size),
                                 np.arange(crop_size),
                                 np.arange(crop_size), indexing="ij")
        self._cell_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()],
                                  axis=-1).astype(np.int64)               # (D³, 3)
        self._cell_coords = ((self._cell_idx.astype(np.float32) + 0.5)
                             / crop_size).astype(np.float32)              # (D³, 3)
        # pre-compute static masks (depend only on D, buf)
        self._loss_mask = edge_buffer_mask(crop_size, self.buf).numpy()
        self._pt_mask = ((self._cell_idx >= self.buf) &
                         (self._cell_idx < crop_size - self.buf)
                         ).all(axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.crops)

    def _build_env(self, env_n: np.ndarray, ext_vox: tuple[int, int, int],
                   crop_origin: tuple[int, int, int]) -> np.ndarray:
        """Apply outside-only masking + append indicator channel if enabled."""
        if not self.env_outside_mask:
            return env_n
        outside = outside_mask_for_crop(
            env_resolution=env_n.shape[-1],
            sim_extent_vox=ext_vox,
            crop_origin_vox=crop_origin,
            crop_side_vox=self.D,
        )                                                                 # (R, R, R)
        # zero out the inside region of every channel
        masked = env_n * outside[None]                                    # (3, R, R, R)
        return np.concatenate([masked, outside[None]], axis=0)            # (4, R, R, R)

    def __getitem__(self, idx: int) -> dict:
        sid, sx, sy, sz, ext_vox = self.crops[idx]
        D = self.D

        lf = self.reader.load_crop(self.lf_root, sid, (sx, sy, sz),
                                   D, ext_vox, self.snapshot)
        hf = self.reader.load_crop(self.hf_root, sid, (sx, sy, sz),
                                   D, ext_vox, self.snapshot)
        lf_n = self.norm.normalize(lf).astype(np.float32)
        hf_n = self.norm.normalize(hf).astype(np.float32)
        residual = (hf_n - lf_n).astype(np.float32)

        env_path = os.path.join(self.st_root, f"set{sid}_quijotelike",
                                self.snapshot, "disp.npy")
        env_raw = self.reader.load_full(env_path)
        env_n = self.norm.normalize(env_raw).astype(np.float32)
        env_n = self._build_env(env_n, ext_vox, (sx, sy, sz))

        style_path = os.path.join(self.hf_root, f"set{sid}_pos_0_0_0",
                                  self.snapshot, "style.npy")
        style = self.reader.load_full(style_path).astype(np.float32)

        # all D³ Lagrangian cells as points (no subsampling)
        i0, i1, i2 = self._cell_idx[:, 0], self._cell_idx[:, 1], self._cell_idx[:, 2]
        lf_pt  = lf_n[:, i0, i1, i2].T.astype(np.float32)                 # (D³, 3)
        tgt_pt = residual[:, i0, i1, i2].T.astype(np.float32)             # (D³, 3)

        return {
            "lf_voxel": torch.from_numpy(lf_n),
            "env":      torch.from_numpy(env_n),
            "coords":   torch.from_numpy(self._cell_coords),
            "lf_pt":    torch.from_numpy(lf_pt),
            "tgt_pt":   torch.from_numpy(tgt_pt),
            "tgt_vox":  torch.from_numpy(residual),
            "loss_mask":torch.from_numpy(self._loss_mask),
            "pt_mask":  torch.from_numpy(self._pt_mask),
            "style":    torch.from_numpy(style),
            "set_id":   sid,
            "crop_xyz": (sx, sy, sz),
            "extent":   ext_vox,
        }
