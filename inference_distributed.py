#!/usr/bin/env python3
"""Distributed inference for the Single-step Point-Voxel Flow-Matching model.

This script reconstructs a high-fidelity (HF) Lagrangian displacement field
from a paired low-fidelity (LF) simulation by tiling the global ``D x D x D``
volume into overlapping patches of side ``L`` (overlap ``d``), running the
trained Point-Voxel flow matcher on each patch in parallel across GPUs, and
stitching the per-patch predictions back into a global HDF5 master file.

The stitch is **lossless** in the sense that every Lagrangian cell of the
global volume is owned by exactly one patch — the cut between two overlapping
patches is placed at the integer midpoint of their overlap region (which
equals ``patch_origin + L - d/2`` for the regular interior overlaps the
model was trained on). Faces of patches that touch the global boundary are
not trimmed, so the reconstruction is bit-exact gap-free.

Launch (single node, 4 GPUs)::

    torchrun --standalone --nproc_per_node=4 inference_distributed.py \\
        --ckpt_path runs/pvfm_a100_<JOB>/ckpt_latest.pt \\
        --lf_input_path /data/group_data/universedata/lagrangian_output_64 \\
        --set_id 9 \\
        --L 32 --d 8 \\
        --output_path /scratch/hf_set9.h5
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist

from data.normalization import NormStats
from data.readers import TileReader, get_reader
from data.simulation_dataset import SNAPSHOT_DEFAULT, discover_sets
from engine.checkpoint import CheckpointManager
from engine.flow_matching import euler_sample
from models import PVFlowMatcher
from ops.geometry import outside_mask_for_crop, overlap_crop_starts


# ---------------------------------------------------------------------------
# Distributed bootstrapping
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize the NCCL process group from torchrun env vars.

    Returns:
        Tuple ``(rank, local_rank, world_size, device)``. Falls back to
        single-process mode if ``RANK`` is not set.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank, local_rank, world_size = 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA GPUs.")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def setup_logger(rank: int) -> logging.Logger:
    """Rank-0-only INFO logging; other ranks are silenced to WARNING."""
    logger = logging.getLogger("pvfm.infer")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s [rank {rank}] %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    return logger


# ---------------------------------------------------------------------------
# Spatial partitioning + per-axis ownership
# ---------------------------------------------------------------------------

def axis_split_bounds(starts: Sequence[int], L: int, L_axis: int) -> List[int]:
    """Per-axis ownership boundaries between adjacent crops along one axis.

    Given crop starts ``s_0 < s_1 < ... < s_{K-1}`` (each crop has length
    ``L``), this returns a list ``b`` of length ``K + 1`` such that crop
    ``k`` owns voxel indices ``[b[k], b[k+1])`` along this axis. Adjacent
    crops are split at the integer midpoint of their overlap region — for
    regular interior overlaps of width ``d``, this places the cut at
    ``s_k + L - d/2``, exactly matching the training-time inner cube.

    Args:
        starts: Crop start indices along this axis (output of
            :func:`overlap_crop_starts`).
        L: Crop side length in voxels.
        L_axis: Length of the global volume along this axis (voxels).

    Returns:
        ``b[0] = 0``, ``b[K] = L_axis``, and ``b[k]`` for ``0 < k < K`` is
        the integer midpoint of the overlap between crops ``k-1`` and ``k``.

    Raises:
        ValueError: if two adjacent crops are non-overlapping.
    """
    K = len(starts)
    if K == 0:
        raise ValueError("starts must be non-empty")
    if K == 1:
        return [0, L_axis]
    bounds: List[int] = [0]
    for k in range(K - 1):
        s_next = starts[k + 1]
        e_curr = starts[k] + L
        if s_next >= e_curr:
            raise ValueError(
                f"adjacent crops do not overlap: end[{k}]={e_curr} <= "
                f"start[{k+1}]={s_next}"
            )
        bounds.append((s_next + e_curr) // 2)
    bounds.append(L_axis)
    return bounds


def enumerate_boxes(D: int, L: int, d: int
                    ) -> List[Dict[str, Tuple[int, int, int] | Tuple[Tuple[int, int], ...]]]:
    """Build the global list of patch bounding boxes + ownership ranges.

    Sliding-window stride is exactly ``L - d`` for interior crops; the last
    crop along each axis is anchored at ``L_axis - L`` to cover the trailing
    edge (handled by :func:`overlap_crop_starts`).

    Args:
        D: Side length of the full volume (voxels).
        L: Patch side length (voxels).
        d: Inter-patch overlap (voxels). Must be even and ``0 < d < L``.

    Returns:
        A list of dicts with keys:
          ``origin``  — ``(sx, sy, sz)`` voxel origin of the patch.
          ``axis_inner`` — ``((ax_lo, ax_hi), (ay_lo, ay_hi), (az_lo, az_hi))``
              the global voxel ranges this patch *owns* (disjoint across
              patches, union covers ``[0, D)^3``).
          ``k``       — ``(kx, ky, kz)`` integer index of the patch.
    """
    if d % 2 != 0:
        raise ValueError(f"--d must be even (it sets buffer = d/2); got {d}")
    if not 0 < d < L:
        raise ValueError(f"need 0 < d < L; got d={d}, L={L}")
    if L > D:
        raise ValueError(f"L={L} cannot exceed D={D}")

    starts = overlap_crop_starts(D, L, d)
    bounds = axis_split_bounds(starts, L, D)

    boxes: List[Dict] = []
    for kx, sx in enumerate(starts):
        for ky, sy in enumerate(starts):
            for kz, sz in enumerate(starts):
                boxes.append({
                    "k": (kx, ky, kz),
                    "origin": (sx, sy, sz),
                    "axis_inner": (
                        (bounds[kx], bounds[kx + 1]),
                        (bounds[ky], bounds[ky + 1]),
                        (bounds[kz], bounds[kz + 1]),
                    ),
                })
    return boxes


def split_indices(n: int, world_size: int, rank: int) -> Tuple[int, int]:
    """Block-partition ``range(n)`` across ranks (last rank takes the tail)."""
    base = n // world_size
    extra = n % world_size
    if rank < extra:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = extra * (base + 1) + (rank - extra) * base
        end = start + base
    return start, end


# ---------------------------------------------------------------------------
# Model + global-context loading
# ---------------------------------------------------------------------------

def build_model_from_ckpt(payload: dict, device: torch.device,
                          use_ema: bool) -> PVFlowMatcher:
    """Reconstruct the model architecture from the checkpoint config and load weights.

    Args:
        payload: Output of :meth:`CheckpointManager.load`.
        device: Target device.
        use_ema: If True and EMA weights are present, load those instead of
            the live weights (recommended for inference).

    Returns:
        A ``PVFlowMatcher`` in eval mode on ``device``.
    """
    cfg = payload["cfg"]
    m = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    default_c_env = 4 if data_cfg.get("env_outside_mask", True) else 3
    model = PVFlowMatcher(
        c_pt=3, c_lf=3, c_env=m.get("c_env", default_c_env), c_lf_pt=3,
        n_style=m.get("n_style", 5),
        base_voxel=m.get("base_voxel", 32),
        base_point=m.get("base_point", 128),
        cond_dim=m.get("cond_dim", 256),
        n_blocks=m.get("n_blocks", 4),
        env_resolution=m.get("env_resolution", 64),
    )
    state = payload["model"]
    if use_ema and payload.get("ema") is not None:
        state = payload["ema"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_global_context(reader: TileReader, root: str, set_id: int,
                        snapshot: str, norm: NormStats
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize the per-set global LF env cube + cosmology vector.

    The "global context" the user refers to is the un-masked normalized LF
    env tensor ``(3, R, R, R)`` — the input to the global-context CNN. We
    keep it un-masked here and apply the per-crop outside mask later (only
    if the model was trained with ``env_outside_mask=True``). The cosmology
    vector is loaded from the HF tile root to match the training pipeline.

    Args:
        reader: The configured tile reader (``numpy`` by default).
        root: Top-level data root (containing ``stitched/``, ``quijote-64/``).
        set_id: Simulation set id.
        snapshot: Snapshot subdirectory.
        norm: Per-channel normalization stats from the checkpoint.

    Returns:
        ``(env_n, style)`` where ``env_n`` is ``(3, R, R, R)`` float32 in
        normalized space and ``style`` is ``(n_style,)`` float32.
    """
    env_path = os.path.join(root, "stitched", f"set{set_id}_quijotelike",
                            snapshot, "disp.npy")
    env_raw = reader.load_full(env_path)
    env_n = norm.normalize(env_raw).astype(np.float32)
    style_path = os.path.join(root, "quijote-64", f"set{set_id}_pos_0_0_0",
                              snapshot, "style.npy")
    style = reader.load_full(style_path).astype(np.float32)
    return env_n, style


def broadcast_array(arr: Optional[np.ndarray], src: int,
                    device: torch.device) -> np.ndarray:
    """Broadcast a numpy array from ``src`` to all ranks via NCCL.

    The shape and dtype are negotiated explicitly so non-source ranks can
    pre-allocate the receive buffer.
    """
    if not is_dist():
        assert arr is not None
        return arr

    rank = dist.get_rank()
    # Negotiate shape (4 ints: ndim + up to 3 dims; clamp to 4 dims for env+style).
    if rank == src:
        assert arr is not None
        ndim = arr.ndim
        shape = list(arr.shape) + [0] * (4 - arr.ndim)
        meta = torch.tensor([ndim] + shape, dtype=torch.long, device=device)
    else:
        meta = torch.zeros(5, dtype=torch.long, device=device)
    dist.broadcast(meta, src=src)
    ndim = int(meta[0].item())
    shape = tuple(int(x) for x in meta[1:1 + ndim].tolist())

    if rank == src:
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)
    else:
        t = torch.empty(shape, dtype=torch.float32, device=device)
    dist.broadcast(t, src=src)
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-patch generation
# ---------------------------------------------------------------------------

def build_env_for_crop(env_n_global: np.ndarray, env_outside_mask: bool,
                       sim_extent_vox: Tuple[int, int, int],
                       crop_origin: Tuple[int, int, int],
                       L: int) -> np.ndarray:
    """Replicate the dataset's env construction for a single crop.

    With ``env_outside_mask=True`` the 3-channel env is zeroed inside the
    crop and a 4-th binary indicator channel (1 outside, 0 inside) is
    appended — exactly matching ``SimulationDataset._build_env``.
    """
    if not env_outside_mask:
        return env_n_global
    R = env_n_global.shape[-1]
    outside = outside_mask_for_crop(
        env_resolution=R,
        sim_extent_vox=sim_extent_vox,
        crop_origin_vox=crop_origin,
        crop_side_vox=L,
    )                                                        # (R, R, R)
    masked = env_n_global * outside[None]                    # (3, R, R, R)
    return np.concatenate([masked, outside[None]], axis=0)   # (4, R, R, R)


def owned_local_indices(L: int, origin: Tuple[int, int, int],
                        axis_inner: Tuple[Tuple[int, int], ...]
                        ) -> np.ndarray:
    """Indices of cells (local to a crop) that this crop owns globally.

    Args:
        L: Crop side length (voxels).
        origin: ``(sx, sy, sz)`` global origin of the crop.
        axis_inner: Per-axis owned global voxel ranges
            ``((ax_lo, ax_hi), (ay_lo, ay_hi), (az_lo, az_hi))``.

    Returns:
        ``(N_own, 3)`` int64 array of local cell indices in ``[0, L)^3``.
    """
    sx, sy, sz = origin
    (ax_lo, ax_hi), (ay_lo, ay_hi), (az_lo, az_hi) = axis_inner
    lx_lo, lx_hi = ax_lo - sx, ax_hi - sx
    ly_lo, ly_hi = ay_lo - sy, ay_hi - sy
    lz_lo, lz_hi = az_lo - sz, az_hi - sz
    # Sanity: ranges must be in [0, L].
    for lo, hi in ((lx_lo, lx_hi), (ly_lo, ly_hi), (lz_lo, lz_hi)):
        if not (0 <= lo < hi <= L):
            raise RuntimeError(
                f"owned local range [{lo}, {hi}) is out of [0, {L}]"
            )
    ii, jj, kk = np.meshgrid(
        np.arange(lx_lo, lx_hi, dtype=np.int64),
        np.arange(ly_lo, ly_hi, dtype=np.int64),
        np.arange(lz_lo, lz_hi, dtype=np.int64),
        indexing="ij",
    )
    return np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1)


@torch.no_grad()
def generate_patch(model: PVFlowMatcher, *,
                   lf_voxel_t: torch.Tensor, env_t: torch.Tensor,
                   style_t: torch.Tensor, coords_t: torch.Tensor,
                   lf_pt_t: torch.Tensor, steps: int,
                   chunk_points: int) -> torch.Tensor:
    """Run K-step Euler sampling over one crop, possibly chunked over points.

    Args:
        model: The trained ``PVFlowMatcher``.
        lf_voxel_t: ``(1, 3, L, L, L)``.
        env_t: ``(1, c_env, R, R, R)``.
        style_t: ``(1, n_style)``.
        coords_t: ``(1, N_own, 3)`` normalized cell-centre coords in [0, 1].
        lf_pt_t: ``(1, N_own, 3)`` normalized LF disp at those cells.
        steps: Number of Euler integration steps (``cfg["flow"]["n_steps_infer"]``).
        chunk_points: Maximum points per forward (limits peak activations).

    Returns:
        ``(N_own, 3)`` predicted normalized residual (HF_n - LF_n).
    """
    N = coords_t.shape[1]
    if chunk_points <= 0 or chunk_points >= N:
        out = euler_sample(model, lf_voxel_t, env_t, style_t,
                           coords_t, lf_pt_t, steps=steps)
        return out.squeeze(0)

    parts: List[torch.Tensor] = []
    for s in range(0, N, chunk_points):
        e = min(s + chunk_points, N)
        out = euler_sample(model, lf_voxel_t, env_t, style_t,
                           coords_t[:, s:e], lf_pt_t[:, s:e], steps=steps)
        parts.append(out.squeeze(0))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# I/O — per-rank temp files + rank-0 merge
# ---------------------------------------------------------------------------

def temp_file_path(temp_dir: Path, rank: int) -> Path:
    return temp_dir / f"temp_hf_volume_rank_{rank:04d}.h5"


def write_rank_temp(path: Path, *, q_idx: np.ndarray, hf_disp: np.ndarray,
                    positions: np.ndarray, set_id: int, rank: int) -> None:
    """Write this rank's collected HF points to its temp HDF5 file.

    Datasets are written with ``compression="lzf"`` to keep the on-disk size
    manageable when the global volume has tens of millions of cells.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["set_id"] = int(set_id)
        f.attrs["rank"] = int(rank)
        f.attrs["n_points"] = int(q_idx.shape[0])
        f.create_dataset("q_idx", data=q_idx, compression="lzf")
        f.create_dataset("hf_disp", data=hf_disp, compression="lzf")
        f.create_dataset("positions", data=positions, compression="lzf")


def merge_temp_files(temp_dir: Path, world_size: int, output_path: Path,
                     *, D: int, L: int, d: int, set_id: int,
                     box_size: float, snapshot: str,
                     logger: logging.Logger) -> None:
    """Concatenate per-rank temp HDF5 files into a contiguous master file.

    Streams from each temp file into the master with a fixed chunk size so
    we never have to materialize the full point set in host RAM. Cleans up
    the temp files on success.
    """
    n_total = 0
    for r in range(world_size):
        p = temp_file_path(temp_dir, r)
        if not p.exists():
            raise FileNotFoundError(f"Missing temp file from rank {r}: {p}")
        with h5py.File(p, "r") as f:
            n_total += int(f["q_idx"].shape[0])
    logger.info("Merging %d temp files (%d total points) -> %s",
                world_size, n_total, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_chunk = 1 << 20  # 1 M points per copy step

    with h5py.File(output_path, "w") as out:
        out.attrs["D"] = int(D)
        out.attrs["L"] = int(L)
        out.attrs["d"] = int(d)
        out.attrs["set_id"] = int(set_id)
        out.attrs["box_size"] = float(box_size)
        out.attrs["snapshot"] = snapshot

        d_qidx = out.create_dataset("q_idx", shape=(n_total, 3),
                                    dtype=np.int32, chunks=True,
                                    compression="lzf")
        d_disp = out.create_dataset("hf_disp", shape=(n_total, 3),
                                    dtype=np.float32, chunks=True,
                                    compression="lzf")
        d_pos = out.create_dataset("positions", shape=(n_total, 3),
                                   dtype=np.float32, chunks=True,
                                   compression="lzf")

        cursor = 0
        for r in range(world_size):
            p = temp_file_path(temp_dir, r)
            with h5py.File(p, "r") as f:
                n_r = int(f["q_idx"].shape[0])
                if n_r == 0:
                    continue
                for s in range(0, n_r, write_chunk):
                    e = min(s + write_chunk, n_r)
                    d_qidx[cursor + s:cursor + e] = f["q_idx"][s:e]
                    d_disp[cursor + s:cursor + e] = f["hf_disp"][s:e]
                    d_pos[cursor + s:cursor + e] = f["positions"][s:e]
                cursor += n_r

        if cursor != n_total:
            raise RuntimeError(f"Merge mismatch: wrote {cursor}, expected {n_total}")

    for r in range(world_size):
        p = temp_file_path(temp_dir, r)
        try:
            p.unlink()
        except OSError:
            logger.warning("Could not remove temp file %s", p)
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        try:
            temp_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-GPU inference for the Single-step PVFM upsampler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to a ckpt_*.pt file (saved by CheckpointManager).")
    p.add_argument("--lf_input_path", type=str, required=True,
                   help="Top-level data root containing quijote-64/, "
                        "quijotelike-64/, stitched/.")
    p.add_argument("--output_path", type=str, required=True,
                   help="Path to the master HDF5 to write.")
    p.add_argument("--set_id", type=int, required=True,
                   help="Simulation set id to upsample.")
    p.add_argument("--snapshot", type=str, default=SNAPSHOT_DEFAULT,
                   help="Snapshot subdirectory.")
    p.add_argument("--D", type=int, default=None,
                   help="Global volume side in voxels. If omitted, derived "
                        "from the on-disk extent of --set_id.")
    p.add_argument("--L", type=int, default=None,
                   help="Patch side in voxels. Defaults to cfg.data.crop_size.")
    p.add_argument("--d", type=int, default=None,
                   help="Patch overlap in voxels (must be even). Defaults to "
                        "cfg.data.crop_overlap.")
    p.add_argument("--n", type=int, default=None,
                   help="Max points per model forward (chunking). Defaults "
                        "to L^3 = no chunking.")
    p.add_argument("--steps", type=int, default=None,
                   help="Number of Euler integration steps. Defaults to "
                        "cfg.flow.n_steps_infer.")
    p.add_argument("--use_ema", action="store_true",
                   help="Load EMA weights from the checkpoint (recommended).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; per-crop noise = manual_seed(seed + crop_idx).")
    p.add_argument("--temp_dir", type=str, default=None,
                   help="Directory for per-rank temp files. Defaults to "
                        "<output_path>.tmp/")
    p.add_argument("--box_size", type=float, default=None,
                   help="Box size (Mpc/h) for periodic-wrapped positions. "
                        "Defaults to cfg.data.box_size.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    logger = setup_logger(rank)

    if rank == 0:
        logger.info("World size: %d  |  device: %s", world_size, device)
        logger.info("Loading checkpoint: %s", args.ckpt_path)

    payload = CheckpointManager.load(args.ckpt_path, map_location="cpu")
    cfg = payload["cfg"]
    norm = NormStats.from_dict(payload["norm"])

    # Resolve hyper-parameters: CLI overrides cfg defaults.
    L = args.L if args.L is not None else int(cfg["data"]["crop_size"])
    d = args.d if args.d is not None else int(cfg["data"]["crop_overlap"])
    steps = args.steps if args.steps is not None \
        else int(cfg.get("flow", {}).get("n_steps_infer", 1))
    box_size = args.box_size if args.box_size is not None \
        else float(cfg.get("data", {}).get("box_size", 1000.0))
    env_outside_mask = bool(cfg.get("data", {}).get("env_outside_mask", True))
    snapshot = args.snapshot

    # Storage backend (numpy / hdf5).
    reader = get_reader(cfg.get("data", {}).get("reader", "numpy"))

    # Discover the on-disk extent of the requested set; this is also our
    # default for --D if the user didn't override it.
    if rank == 0:
        sets = discover_sets(args.lf_input_path, reader, snapshot)
        match = [s for s in sets if s[0] == args.set_id]
        if not match:
            raise RuntimeError(
                f"set_id={args.set_id} not found (or incomplete) under "
                f"{args.lf_input_path}"
            )
        ext_tiles = match[0][1]
        ext_vox = (ext_tiles[0] * reader.tile_size,
                   ext_tiles[1] * reader.tile_size,
                   ext_tiles[2] * reader.tile_size)
        ext_payload = np.array(ext_vox, dtype=np.int64)
    else:
        ext_payload = np.zeros(3, dtype=np.int64)

    if is_dist():
        ext_t = torch.from_numpy(ext_payload).to(device)
        dist.broadcast(ext_t, src=0)
        ext_payload = ext_t.cpu().numpy()
    ext_vox = (int(ext_payload[0]), int(ext_payload[1]), int(ext_payload[2]))

    if args.D is None:
        if not (ext_vox[0] == ext_vox[1] == ext_vox[2]):
            raise RuntimeError(
                f"Set {args.set_id} has non-cubic extent {ext_vox}; please "
                f"pass --D explicitly."
            )
        D = ext_vox[0]
    else:
        D = int(args.D)
        if D > min(ext_vox):
            raise RuntimeError(
                f"--D={D} exceeds the on-disk extent {ext_vox} for set "
                f"{args.set_id}."
            )

    if L != int(cfg["data"]["crop_size"]) or d != int(cfg["data"]["crop_overlap"]):
        if rank == 0:
            logger.warning(
                "Using L=%d, d=%d which differ from training "
                "(crop_size=%d, crop_overlap=%d). The trained weights may "
                "not generalize.",
                L, d, cfg["data"]["crop_size"], cfg["data"]["crop_overlap"],
            )

    # Build model + load (optionally EMA) weights on every rank.
    model = build_model_from_ckpt(payload, device, args.use_ema)

    # Free the (CPU-side) checkpoint payload — large state dicts otherwise
    # double our resident memory once weights live on device.
    del payload

    # ----- Global context: load on rank 0, broadcast to all -----
    if rank == 0:
        env_n_global, style_np = load_global_context(
            reader, args.lf_input_path, args.set_id, snapshot, norm
        )
        logger.info(
            "Global env shape: %s  |  style shape: %s  |  D=%d, L=%d, d=%d, "
            "buffer=d/2=%d, env_outside_mask=%s, steps=%d",
            env_n_global.shape, style_np.shape, D, L, d, d // 2,
            env_outside_mask, steps,
        )
    else:
        env_n_global = None
        style_np = None

    env_n_global = broadcast_array(env_n_global, src=0, device=device)
    style_np = broadcast_array(style_np, src=0, device=device)
    style_t = torch.from_numpy(style_np).to(device).unsqueeze(0)  # (1, n_style)

    # ----- Bounding boxes + per-rank assignment -----
    boxes = enumerate_boxes(D, L, d)
    n_boxes = len(boxes)
    s_idx, e_idx = split_indices(n_boxes, world_size, rank)
    my_boxes = list(enumerate(boxes))[s_idx:e_idx]
    if rank == 0:
        logger.info("Total patches: %d  |  per-rank: %d–%d patches",
                    n_boxes, n_boxes // world_size, (n_boxes + world_size - 1) // world_size)

    chunk_points = args.n if args.n is not None else (L ** 3)
    voxel_size = box_size / float(D)  # Mpc/h per voxel

    # Per-rank accumulators (host arrays) — cleared via concatenation at end.
    q_idx_buf: List[np.ndarray] = []
    disp_buf: List[np.ndarray] = []
    pos_buf: List[np.ndarray] = []

    lf_root = os.path.join(args.lf_input_path, "quijotelike-64")

    t_start = time.time()
    n_local = 0
    n_local_total = len(my_boxes)
    for j, (gidx, box) in enumerate(my_boxes):
        origin = box["origin"]
        axis_inner = box["axis_inner"]
        sx, sy, sz = origin

        # 1) Load + normalize the LF crop.
        lf = reader.load_crop(lf_root, args.set_id, origin, L, ext_vox, snapshot)
        lf_n = norm.normalize(lf).astype(np.float32)                  # (3, L, L, L)
        lf_voxel_t = torch.from_numpy(lf_n).unsqueeze(0).to(device)   # (1, 3, L, L, L)

        # 2) Build the per-crop env (outside-mask + indicator if trained that way).
        env_in = build_env_for_crop(env_n_global, env_outside_mask,
                                    ext_vox, origin, L)
        env_t = torch.from_numpy(env_in).unsqueeze(0).to(device)

        # 3) Owned (local) cell indices — these globally tile [0, D)^3 exactly once.
        cell_idx = owned_local_indices(L, origin, axis_inner)         # (N_own, 3)
        N_own = cell_idx.shape[0]
        if N_own == 0:
            continue

        coords = (cell_idx.astype(np.float32) + 0.5) / float(L)       # (N_own, 3) in [0, 1]
        ix, iy, iz = cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2]
        lf_pt = lf_n[:, ix, iy, iz].T.astype(np.float32)              # (N_own, 3)

        coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)
        lf_pt_t = torch.from_numpy(lf_pt).unsqueeze(0).to(device)

        # 4) Generate. Per-crop seed → reproducible regardless of #ranks.
        torch.manual_seed(args.seed + gidx)
        torch.cuda.manual_seed_all(args.seed + gidx)
        pred_residual_n = generate_patch(
            model,
            lf_voxel_t=lf_voxel_t, env_t=env_t, style_t=style_t,
            coords_t=coords_t, lf_pt_t=lf_pt_t,
            steps=steps, chunk_points=chunk_points,
        )                                                             # (N_own, 3) normalized

        # 5) Reconstruct HF displacement in physical units.
        hf_disp_n = lf_pt_t.squeeze(0) + pred_residual_n              # normalized
        hf_disp_n_np = hf_disp_n.detach().cpu().numpy().astype(np.float32)
        # NormStats stores per-channel mean/std; broadcast over points.
        hf_disp = (hf_disp_n_np * norm.std.reshape(1, 3) + norm.mean.reshape(1, 3)
                   ).astype(np.float32)

        # 6) Global Lagrangian voxel index + final position (periodic-wrapped).
        q_idx_global = (cell_idx + np.array(origin, dtype=np.int64)).astype(np.int32)
        q_phys = (q_idx_global.astype(np.float32) + 0.5) * voxel_size
        positions = np.mod(q_phys + hf_disp, box_size).astype(np.float32)

        q_idx_buf.append(q_idx_global)
        disp_buf.append(hf_disp)
        pos_buf.append(positions)

        n_local += N_own
        if rank == 0 and ((j + 1) % max(1, n_local_total // 20) == 0
                          or j == n_local_total - 1):
            dt = time.time() - t_start
            logger.info("rank0 progress: %d/%d patches  |  %d points  |  %.1fs",
                        j + 1, n_local_total, n_local, dt)

    # ----- Per-rank concat + temp HDF5 write -----
    if q_idx_buf:
        q_idx_arr = np.concatenate(q_idx_buf, axis=0)
        disp_arr = np.concatenate(disp_buf, axis=0)
        pos_arr = np.concatenate(pos_buf, axis=0)
    else:
        q_idx_arr = np.zeros((0, 3), dtype=np.int32)
        disp_arr = np.zeros((0, 3), dtype=np.float32)
        pos_arr = np.zeros((0, 3), dtype=np.float32)
    # Free intermediate buffers before the heavy h5 write.
    q_idx_buf.clear(); disp_buf.clear(); pos_buf.clear()

    output_path = Path(args.output_path)
    temp_dir = Path(args.temp_dir) if args.temp_dir else \
        output_path.parent / (output_path.name + ".tmp")
    if rank == 0:
        temp_dir.mkdir(parents=True, exist_ok=True)
    barrier()  # ensure temp_dir exists everywhere before writing

    write_rank_temp(temp_file_path(temp_dir, rank),
                    q_idx=q_idx_arr, hf_disp=disp_arr, positions=pos_arr,
                    set_id=args.set_id, rank=rank)

    # Hard sync before merge.
    barrier()

    if rank == 0:
        merge_temp_files(temp_dir, world_size, output_path,
                         D=D, L=L, d=d, set_id=args.set_id,
                         box_size=box_size, snapshot=snapshot, logger=logger)
        logger.info("Done in %.1fs. Master HDF5: %s",
                    time.time() - t_start, output_path)

    cleanup_distributed()


if __name__ == "__main__":
    main()
