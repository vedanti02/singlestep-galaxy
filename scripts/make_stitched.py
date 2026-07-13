#!/usr/bin/env python3
"""Regenerate the ``stitched/`` global LF env cubes.

The training pipeline conditions on a per-set "env" cube: the whole-box
low-fidelity displacement field block-mean downsampled to a fixed
``(3, R, R, R)`` grid (R = ``model.env_resolution`` = 64). The original
``stitched/`` directory was deleted from the shared data root, and that
root is not writable, so this script rebuilds the cubes into a NEW root
that contains symlinks to the original ``quijote-64/`` and
``quijotelike-64/`` tile trees plus a freshly generated ``stitched/``.

Layout written (matching data/factory.py and inference_distributed.py):

    <dst_root>/quijote-64      -> symlink to <src_root>/quijote-64
    <dst_root>/quijotelike-64  -> symlink to <src_root>/quijotelike-64
    <dst_root>/stitched/set{sid}_quijotelike/{snapshot}/disp.npy   (3, R, R, R) float32

Block-mean pooling handles non-cubic extents per axis (a set of tile
extent (ex, ey, ez) has full resolution (64*ex, 64*ey, 64*ez); each env
voxel averages an (ex, ey, ez) block), matching the per-axis R/L ratio
mapping in ops.geometry.outside_mask_for_crop.

Usage:
    python scripts/make_stitched.py \
        --src_root /data/group_data/universedata/lagrangian_output_64 \
        --dst_root /data/user_data/vkshirsa/lagrangian_output_64_fixed \
        --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.readers import get_reader  # noqa: E402
from data.simulation_dataset import _is_full_box  # noqa: E402

TILE = 64


def discover_complete_sets(root: str) -> list[tuple[int, tuple[int, int, int]]]:
    """All set ids with complete, extent-matched HF and LF tile boxes.

    Same completeness rule as ``data.simulation_dataset.discover_sets``
    (via the shared ``_is_full_box``) minus the stitched/ existence check
    — this script is what creates stitched/ in the first place.
    """
    reader = get_reader("numpy")
    hf = reader.index_root(os.path.join(root, "quijote-64"))
    lf = reader.index_root(os.path.join(root, "quijotelike-64"))

    out = []
    for sid in sorted(set(hf) & set(lf)):
        fh, eh = _is_full_box(hf[sid])
        fl, el = _is_full_box(lf[sid])
        if fh and fl and eh == el:
            out.append((sid, eh))
    return out


def build_env_cube(src_root: str, sid: int, ext: tuple[int, int, int],
                   snapshot: str, R: int) -> np.ndarray:
    """Assemble the full-res LF disp grid and block-mean pool to (3, R, R, R)."""
    ex, ey, ez = ext
    Lx, Ly, Lz = ex * TILE, ey * TILE, ez * TILE
    if Lx % R or Ly % R or Lz % R:
        raise ValueError(f"set {sid}: extent {ext} not divisible into R={R}")
    fx, fy, fz = Lx // R, Ly // R, Lz // R

    lf_root = os.path.join(src_root, "quijotelike-64")
    full = np.empty((3, Lx, Ly, Lz), dtype=np.float32)
    for ix in range(ex):
        for iy in range(ey):
            for iz in range(ez):
                p = os.path.join(lf_root, f"set{sid}_pos_{ix}_{iy}_{iz}",
                                 snapshot, "disp.npy")
                full[:, ix * TILE:(ix + 1) * TILE,
                        iy * TILE:(iy + 1) * TILE,
                        iz * TILE:(iz + 1) * TILE] = np.load(p)
    # Block-mean pool: (3, R, fx, R, fy, R, fz) -> mean over the block axes.
    pooled = full.reshape(3, R, fx, R, fy, R, fz).mean(axis=(2, 4, 6))
    return pooled.astype(np.float32)


def _worker(job: tuple) -> tuple[int, str]:
    src_root, dst_root, sid, ext, snapshot, R, skip = job
    out_dir = os.path.join(dst_root, "stitched", f"set{sid}_quijotelike", snapshot)
    out_path = os.path.join(out_dir, "disp.npy")
    if skip and os.path.exists(out_path):
        return sid, "skipped"
    try:
        cube = build_env_cube(src_root, sid, ext, snapshot, R)
        os.makedirs(out_dir, exist_ok=True)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, cube)
        os.replace(tmp, out_path)
        return sid, "ok"
    except Exception as e:  # keep going; report at the end
        return sid, f"ERROR: {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", type=str,
                   default="/data/group_data/universedata/lagrangian_output_64")
    p.add_argument("--dst_root", type=str, required=True)
    p.add_argument("--snapshot", type=str, default="PART_009")
    p.add_argument("--env_resolution", type=int, default=64)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max_sets", type=int, default=0,
                   help="cap for smoke testing (0 = all)")
    p.add_argument("--no_skip_existing", action="store_true")
    args = p.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    for sub in ("quijote-64", "quijotelike-64"):
        link = os.path.join(args.dst_root, sub)
        target = os.path.join(args.src_root, sub)
        if not os.path.exists(link):
            os.symlink(target, link)
        elif os.path.realpath(link) != os.path.realpath(target):
            raise RuntimeError(f"{link} exists and does not point to {target}")

    sets = discover_complete_sets(args.src_root)
    if args.max_sets:
        sets = sets[:args.max_sets]
    print(f"[make_stitched] {len(sets)} complete sets -> "
          f"{args.dst_root}/stitched (R={args.env_resolution})", flush=True)

    jobs = [(args.src_root, args.dst_root, sid, ext, args.snapshot,
             args.env_resolution, not args.no_skip_existing)
            for sid, ext in sets]
    t0 = time.time()
    n_ok = n_skip = 0
    errors = []
    with Pool(args.workers) as pool:
        for i, (sid, status) in enumerate(pool.imap_unordered(_worker, jobs)):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                errors.append((sid, status))
            if (i + 1) % 100 == 0 or (i + 1) == len(jobs):
                print(f"[make_stitched] {i + 1}/{len(jobs)}  ok={n_ok} "
                      f"skip={n_skip} err={len(errors)}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    for sid, msg in errors:
        print(f"[make_stitched] set {sid}: {msg}", flush=True)
    if errors:
        sys.exit(1)
    print(f"[make_stitched] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
