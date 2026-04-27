"""Standalone tests for ops/geometry — no torch.nn / no I/O dependencies.

Run with:
    python -m pytest tests/test_geometry.py
or:
    python tests/test_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow `python tests/test_geometry.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ops.geometry import (apply_periodic_bc, edge_buffer_mask, inner_crop,
                          minimum_image, overlap_crop_starts,
                          point_in_inner_mask, points_to_voxel,
                          sample_lagrangian_points, trilinear_devoxelize,
                          voxel_to_points_indices)


def test_inner_crop_strips_buffer():
    g = torch.arange(8 ** 3, dtype=torch.float32).reshape(1, 1, 8, 8, 8)
    inner = inner_crop(g, buffer=2)
    assert inner.shape == (1, 1, 4, 4, 4)
    assert inner[0, 0, 0, 0, 0] == g[0, 0, 2, 2, 2]


def test_edge_buffer_mask_is_one_inside():
    m = edge_buffer_mask(D=8, buffer=2)
    assert m.shape == (8, 8, 8)
    assert m.sum().item() == 4 * 4 * 4
    assert m[0, 0, 0].item() == 0.0
    assert m[4, 4, 4].item() == 1.0


def test_overlap_crop_starts_covers_volume():
    # exact tiling
    assert overlap_crop_starts(L=128, D=64, overlap=0) == [0, 64]
    # uniform stride leaves a tail → final crop anchored to L-D
    starts = overlap_crop_starts(L=128, D=64, overlap=16)
    assert starts[0] == 0 and starts[-1] + 64 == 128
    # crop bigger than volume → single zero start
    assert overlap_crop_starts(L=64, D=128, overlap=16) == [0]


def test_point_in_inner_mask_excludes_buffer():
    coords = torch.tensor([[
        [0.05, 0.5, 0.5],   # x in buffer
        [0.5,  0.5, 0.5],   # inside
        [0.5,  0.95, 0.5],  # y in buffer
    ]])
    m = point_in_inner_mask(coords, buffer_frac=0.1)
    assert m.tolist() == [[0.0, 1.0, 0.0]]


def test_periodic_bc_wraps():
    pos = torch.tensor([[1005.0, -3.0, 0.0]])
    w = apply_periodic_bc(pos, 1000.0)
    assert torch.allclose(w, torch.tensor([[5.0, 997.0, 0.0]]))


def test_minimum_image_picks_shortest_separation():
    delta = torch.tensor([[700.0, -400.0, 100.0]])
    mi = minimum_image(delta, 1000.0)
    assert torch.allclose(mi, torch.tensor([[-300.0, -400.0, 100.0]]))


def test_trilinear_at_cell_centres_matches_indexed_gather():
    B, C, R, N = 2, 3, 16, 64
    grid = torch.randn(B, C, R, R, R)
    idx = torch.randint(0, R, (B, N, 3))
    coords = (idx.float() + 0.5) / R
    a = voxel_to_points_indices(grid, idx)
    b = trilinear_devoxelize(grid, coords)
    assert torch.allclose(a, b, atol=1e-5)


def test_points_to_voxel_roundtrip_mean():
    B, R, N = 1, 8, 256
    coords = torch.rand(B, N, 3)
    feats  = torch.randn(B, N, 3)
    g = points_to_voxel(coords, feats, R, reduction="mean")
    assert g.shape == (B, 3, R, R, R)


def test_sample_lagrangian_points_in_range():
    rng = np.random.default_rng(42)
    idx, coords = sample_lagrangian_points(D=32, n_points=128, rng=rng)
    assert idx.dtype == np.int64 and idx.min() >= 0 and idx.max() < 32
    assert coords.dtype == np.float32 and 0 < coords.min() and coords.max() < 1


if __name__ == "__main__":
    n_pass, n_fail = 0, 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}"); n_pass += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}"); n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
