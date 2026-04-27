"""Tile readers — abstract over storage format (.npy, HDF5).

The dataset only ever asks the reader for the same three things:

1. ``index_root(root)`` — discover which (set_id, tile_xyz) pairs exist.
2. ``load_crop(...)``  — sparse-load a (3, D, D, D) crop.
3. ``load_full(...)``  — load a full per-tile array (used for stitched env).

A factory :func:`get_reader` returns the right reader for a given
``cfg["data"]["reader"]`` string. New formats only have to subclass
:class:`TileReader` and register themselves in :data:`_READERS`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

TILE_SIZE_DEFAULT = 64


class TileReader(ABC):
    """Abstract per-tile storage backend."""

    tile_size: int = TILE_SIZE_DEFAULT

    @abstractmethod
    def index_root(self, root: str) -> dict[int, list[tuple[int, int, int]]]:
        """Return ``{set_id: [(tile_x, tile_y, tile_z), ...]}`` discovered in ``root``."""

    @abstractmethod
    def load_tile(self, root: str, set_id: int,
                  tile_xyz: tuple[int, int, int],
                  snapshot: str, field: str = "disp") -> np.ndarray:
        """Return a single ``(3, tile_size, tile_size, tile_size)`` tile."""

    @abstractmethod
    def load_full(self, path: str) -> np.ndarray:
        """Return a single full array (used for stitched env / style)."""

    # -------- shared sparse-crop logic (axis tiling) --------

    def _axis_tiles(self, start: int, length: int, max_size: int):
        end = min(start + length, max_size)
        pos = start
        while pos < end:
            tile_idx = pos // self.tile_size
            in_tile_start = pos - tile_idx * self.tile_size
            copy_len = min(self.tile_size - in_tile_start, end - pos)
            yield tile_idx, in_tile_start, copy_len, pos - start
            pos += copy_len

    def load_crop(self, root: str, set_id: int,
                  origin_xyz: tuple[int, int, int],
                  D: int,
                  extent_vox: tuple[int, int, int],
                  snapshot: str,
                  field: str = "disp") -> np.ndarray:
        """Sparse-load a ``(3, D, D, D)`` crop touching at most 8 tiles."""
        sx, sy, sz = origin_xyz
        Lx, Ly, Lz = extent_vox
        out = np.zeros((3, D, D, D), dtype=np.float32)
        for ix, srcx, lx, dstx in self._axis_tiles(sx, D, Lx):
            for iy, srcy, ly, dsty in self._axis_tiles(sy, D, Ly):
                for iz, srcz, lz, dstz in self._axis_tiles(sz, D, Lz):
                    arr = self.load_tile(root, set_id, (ix, iy, iz),
                                         snapshot, field)
                    out[:, dstx:dstx + lx, dsty:dsty + ly, dstz:dstz + lz] = \
                        arr[:, srcx:srcx + lx, srcy:srcy + ly, srcz:srcz + lz]
        return out


# ---------------------------------------------------------------------------
# Concrete: numpy-on-disk
# ---------------------------------------------------------------------------

class NumpyTileReader(TileReader):
    """Reader for the on-disk layout: ``set{i}_pos_{x}_{y}_{z}/{snapshot}/{field}.npy``."""

    def index_root(self, root: str) -> dict[int, list[tuple[int, int, int]]]:
        out: dict[int, list[tuple[int, int, int]]] = {}
        for n in os.listdir(root):
            if not n.startswith("set"):
                continue
            try:
                parts = n.split("_")
                sid = int(parts[0][3:])
                x, y, z = int(parts[2]), int(parts[3]), int(parts[4])
            except (ValueError, IndexError):
                continue
            out.setdefault(sid, []).append((x, y, z))
        return out

    def load_tile(self, root: str, set_id: int,
                  tile_xyz: tuple[int, int, int],
                  snapshot: str, field: str = "disp") -> np.ndarray:
        ix, iy, iz = tile_xyz
        p = os.path.join(root, f"set{set_id}_pos_{ix}_{iy}_{iz}",
                         snapshot, f"{field}.npy")
        return np.load(p, mmap_mode="r")

    def load_full(self, path: str) -> np.ndarray:
        return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete: HDF5 (placeholder)
# ---------------------------------------------------------------------------

class HDF5Reader(TileReader):
    """HDF5 reader stub. Implement when an HDF5 layout is provided.

    Expected scheme (subject to revision once an example file lands):
        - One file per set: ``{root}/set{i}.h5``
        - Datasets:    ``/{snapshot}/{field}/tile_{x}_{y}_{z}`` shape (3, 64, 64, 64).
    """

    def __init__(self) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as e:                                 # pragma: no cover
            raise ImportError(
                "HDF5Reader requires h5py — install with `pip install h5py`."
            ) from e

    def index_root(self, root: str) -> dict[int, list[tuple[int, int, int]]]:
        raise NotImplementedError("HDF5 layout not yet specified — provide an "
                                  "example file under /home/vkshirsa to wire this up.")

    def load_tile(self, root: str, set_id: int,
                  tile_xyz: tuple[int, int, int],
                  snapshot: str, field: str = "disp") -> np.ndarray:
        raise NotImplementedError

    def load_full(self, path: str) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_READERS: dict[str, type[TileReader]] = {
    "numpy": NumpyTileReader,
    "hdf5":  HDF5Reader,
}


def get_reader(name: str) -> TileReader:
    """Return a reader instance for ``name`` ('numpy' or 'hdf5').

    Args:
        name: Backend identifier from ``cfg["data"]["reader"]``.

    Returns:
        A fresh :class:`TileReader` instance.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    if name not in _READERS:
        raise ValueError(f"unknown reader '{name}'; choose from {list(_READERS)}")
    return _READERS[name]()
