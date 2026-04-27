"""Evaluator — full-volume inference + comparison plots and JSON dumps.

Mirrors the U-Net baseline outputs (slices, P(k), T(k), coherence,
residual histogram, density panels) so this run can be benchmarked
side-by-side. Each held-out simulation produces:

* ``set{i}_disp_slice.png``       — 2D mean intensity along an axis
* ``set{i}_disp{c}_pk.png``       — log-log P(k) per channel
* ``set{i}_disp_T.png``           — transfer function across channels
* ``set{i}_disp_r.png``           — cross-coherence across channels
* ``set{i}_disp_residual.png``    — voxel-residual histogram
* ``set{i}_density_*.png``        — same trio on the CIC density field
* ``set{i}_stats.json``           — every k-binned stat as raw arrays

Use:

    python eval.py --ckpt runs/pvfm/ckpt_latest.pt --max_sets 3
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data import (NormStats, SNAPSHOT_DEFAULT, get_reader)
from data.simulation_dataset import discover_sets, split_sets
from engine.flow_matching import euler_sample
from models import PVFlowMatcher
from ops import (coherence, disp_to_density, outside_mask_for_crop,
                 overlap_crop_starts, power_spectrum, transfer_function)


# ---------------------------------------------------------------------------
# Helper structs
# ---------------------------------------------------------------------------

@dataclass
class FullVolumeResult:
    """Stitched full-simulation cubes (denormalized)."""
    set_id: int
    extent: tuple[int, int, int]               # tile-extent (e.g. (2,2,2))
    lf:   np.ndarray                           # (3, Lx, Ly, Lz)
    hf:   np.ndarray                           # (3, Lx, Ly, Lz)
    pred: np.ndarray                           # (3, Lx, Ly, Lz)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Run a trained model on the held-out test set and dump plots + JSON.

    Args:
        model: A trained :class:`models.PVFlowMatcher` (in eval mode).
        cfg:   The config used to train it (for crop_size, overlap, etc.).
        norm:  The :class:`NormStats` used during training.
        out_dir: Directory to write outputs to (created if missing).
        steps: Euler-integration steps. ``1`` = single-step (default).
        chunk_pts: Max points per network forward pass (memory budget).
        n_bins: Radial bins for power-spectrum statistics.
    """

    def __init__(self,
                 model: PVFlowMatcher,
                 cfg: Config,
                 norm: NormStats,
                 out_dir: str | Path,
                 steps: int = 1,
                 chunk_pts: int = 131072,
                 n_bins: int = 32) -> None:
        self.model = model.eval()
        self.cfg = cfg
        self.norm = norm
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps = steps
        self.chunk_pts = chunk_pts
        self.n_bins = n_bins

        self.device = next(model.parameters()).device
        self.reader = get_reader(cfg["data"].get("reader", "numpy"))
        self.snapshot = cfg["data"].get("snapshot", SNAPSHOT_DEFAULT)
        self.D = cfg["data"]["crop_size"]
        self.overlap = cfg["data"]["crop_overlap"]
        self.buf = self.overlap // 2
        self.box_size = float(cfg["data"].get("box_size",
                                              self.D))   # for axis units
        self.env_outside_mask = cfg["data"].get("env_outside_mask", True)

    # ------------------------------------------------------------------
    # full-volume inference
    # ------------------------------------------------------------------

    def _predict_crop(self, lf_n: torch.Tensor, env_n: torch.Tensor,
                      style: torch.Tensor) -> torch.Tensor:
        """Predict the normalized residual at every voxel of one crop."""
        D = self.D
        device = lf_n.device
        grid = torch.stack(torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(D, device=device),
            torch.arange(D, device=device), indexing="ij"), dim=-1).reshape(-1, 3)
        coords = (grid.float() + 0.5) / D
        lf_pt_full = lf_n[0, :, grid[:, 0], grid[:, 1], grid[:, 2]].T
        N = coords.shape[0]
        out = torch.empty(N, 3, dtype=lf_n.dtype, device="cpu")
        for s in range(0, N, self.chunk_pts):
            e = min(s + self.chunk_pts, N)
            c_chunk  = coords[s:e].unsqueeze(0)
            lf_chunk = lf_pt_full[s:e].unsqueeze(0)
            pred = euler_sample(self.model, lf_n, env_n, style,
                                c_chunk, lf_chunk, steps=self.steps)
            out[s:e] = pred[0].cpu()
        return out.T.reshape(3, D, D, D)

    def predict_full_volume(self, set_id: int,
                            extent: tuple[int, int, int]) -> FullVolumeResult:
        cfg = self.cfg
        root = cfg["data"]["root"]
        Lx, Ly, Lz = (extent[0] * self.reader.tile_size,
                      extent[1] * self.reader.tile_size,
                      extent[2] * self.reader.tile_size)
        ext_vox = (Lx, Ly, Lz)

        env = self.reader.load_full(os.path.join(
            root, "stitched", f"set{set_id}_quijotelike",
            self.snapshot, "disp.npy"))
        env_n_arr = self.norm.normalize(env).astype(np.float32)  # (3, R, R, R)
        # env_t built per crop below if outside-masking is enabled

        style = self.reader.load_full(os.path.join(
            root, "quijote-64", f"set{set_id}_pos_0_0_0",
            self.snapshot, "style.npy"))
        style_t = torch.from_numpy(style.astype(np.float32)).unsqueeze(0).to(self.device)

        sx_l = overlap_crop_starts(Lx, self.D, self.overlap)
        sy_l = overlap_crop_starts(Ly, self.D, self.overlap)
        sz_l = overlap_crop_starts(Lz, self.D, self.overlap)

        pred_full = np.zeros((3, Lx, Ly, Lz), dtype=np.float32)
        lf_full   = np.zeros_like(pred_full)
        hf_full   = np.zeros_like(pred_full)
        cnt       = np.zeros((Lx, Ly, Lz), dtype=np.float32)

        n = len(sx_l) * len(sy_l) * len(sz_l)
        print(f"  set {set_id} (ext {extent}): {n} crops...")
        k = 0
        for sx in sx_l:
            for sy in sy_l:
                for sz in sz_l:
                    k += 1
                    lf = self.reader.load_crop(os.path.join(root, "quijotelike-64"),
                                               set_id, (sx, sy, sz),
                                               self.D, ext_vox, self.snapshot)
                    hf = self.reader.load_crop(os.path.join(root, "quijote-64"),
                                               set_id, (sx, sy, sz),
                                               self.D, ext_vox, self.snapshot)
                    lf_full[:, sx:sx + self.D, sy:sy + self.D, sz:sz + self.D] = lf
                    hf_full[:, sx:sx + self.D, sy:sy + self.D, sz:sz + self.D] = hf

                    lf_n = self.norm.normalize(lf).astype(np.float32)
                    lf_t = torch.from_numpy(lf_n).unsqueeze(0).to(self.device)
                    if self.env_outside_mask:
                        outside = outside_mask_for_crop(
                            env_resolution=env_n_arr.shape[-1],
                            sim_extent_vox=ext_vox,
                            crop_origin_vox=(sx, sy, sz),
                            crop_side_vox=self.D)
                        env_crop = np.concatenate(
                            [env_n_arr * outside[None], outside[None]],
                            axis=0).astype(np.float32)
                    else:
                        env_crop = env_n_arr
                    env_t = torch.from_numpy(env_crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        res_n = self._predict_crop(lf_t, env_t, style_t).numpy()
                    hf_pred = self.norm.denormalize(lf_n + res_n).astype(np.float32)

                    ix0 = 0      if sx == 0       else self.buf
                    ix1 = self.D if sx + self.D >= Lx else self.D - self.buf
                    iy0 = 0      if sy == 0       else self.buf
                    iy1 = self.D if sy + self.D >= Ly else self.D - self.buf
                    iz0 = 0      if sz == 0       else self.buf
                    iz1 = self.D if sz + self.D >= Lz else self.D - self.buf
                    pred_full[:, sx + ix0:sx + ix1, sy + iy0:sy + iy1,
                              sz + iz0:sz + iz1] += hf_pred[:, ix0:ix1, iy0:iy1, iz0:iz1]
                    cnt[sx + ix0:sx + ix1, sy + iy0:sy + iy1,
                        sz + iz0:sz + iz1] += 1
                    if k % 8 == 0 or k == n:
                        print(f"    crop {k}/{n}")
        cnt = np.maximum(cnt, 1.0)
        pred_full /= cnt[None]
        return FullVolumeResult(set_id=set_id, extent=extent,
                                lf=lf_full, hf=hf_full, pred=pred_full)

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def _slices_plot(self, r: FullVolumeResult, axis: int = 0) -> None:
        titles = ["LF (input)", "HF (truth)", "HF (pred)"]
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        for row, F in enumerate([r.lf, r.hf, r.pred]):
            for c in range(3):
                img = F[c].mean(axis=axis)
                vmax = float(np.percentile(np.abs(img), 99))
                axes[row, c].imshow(img, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
                axes[row, c].set_title(f"{titles[row]} — disp[{c}]")
                axes[row, c].axis("off")
        fig.suptitle(f"set {r.set_id} — mean over axis {axis}")
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_disp_slice.png", dpi=120)
        plt.close(fig)

    def _pk_plot(self, r: FullVolumeResult, channel: int) -> dict:
        fig, ax = plt.subplots(figsize=(7, 5))
        out = {}
        for label, F in [("lf", r.lf), ("hf", r.hf), ("pred", r.pred)]:
            k, Pk = power_spectrum(F[channel], self.box_size, self.n_bins)
            ax.loglog(k[1:], Pk[1:], label=label, lw=1.7)
            out[label] = {"k": k.tolist(), "P": Pk.tolist()}
        ax.set_xlabel(r"$k$"); ax.set_ylabel(f"$P(k)$ disp[{channel}]")
        ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_disp{channel}_pk.png", dpi=120)
        plt.close(fig)
        return out

    def _T_plot(self, r: FullVolumeResult) -> dict:
        fig, ax = plt.subplots(figsize=(7, 5))
        out = {}
        for c in range(3):
            k, T, _, _ = transfer_function(r.pred[c], r.hf[c],
                                           self.box_size, self.n_bins)
            ax.semilogx(k[1:], T[1:], label=f"disp[{c}]", lw=1.7)
            out[f"disp{c}"] = {"k": k.tolist(), "T": T.tolist()}
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$T(k)$")
        ax.set_ylim(0, 1.4); ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_disp_T.png", dpi=120)
        plt.close(fig)
        return out

    def _r_plot(self, r: FullVolumeResult) -> dict:
        fig, ax = plt.subplots(figsize=(7, 5))
        out = {}
        for c in range(3):
            k, rk = coherence(r.pred[c], r.hf[c], self.box_size, self.n_bins)
            ax.semilogx(k[1:], rk[1:], label=f"disp[{c}]", lw=1.7)
            out[f"disp{c}"] = {"k": k.tolist(), "r": rk.tolist()}
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$r(k)$")
        ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_disp_r.png", dpi=120)
        plt.close(fig)
        return out

    def _residual_hist(self, r: FullVolumeResult) -> dict:
        diff = (r.pred - r.hf).ravel()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(diff, bins=200, density=True, alpha=0.8)
        mean = float(np.mean(diff)); med = float(np.median(diff))
        std = float(np.std(diff))
        ax.axvline(0, color="k", lw=1)
        ax.axvline(mean, color="r", ls="--", lw=1, label=f"mean={mean:.4f}")
        ax.axvline(med,  color="g", ls="--", lw=1, label=f"median={med:.4f}")
        ax.set_xlabel("HF_pred − HF_truth (disp)"); ax.set_ylabel("density")
        ax.set_yscale("log"); ax.legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_disp_residual.png", dpi=120)
        plt.close(fig)
        return {"mean": mean, "median": med, "std": std}

    def _density_panels(self, r: FullVolumeResult) -> dict:
        rho_lf   = disp_to_density(r.lf,   self.box_size)
        rho_hf   = disp_to_density(r.hf,   self.box_size)
        rho_pred = disp_to_density(r.pred, self.box_size)

        # P(k)
        fig, ax = plt.subplots(figsize=(7, 5))
        out = {}
        for label, F in [("lf", rho_lf), ("hf", rho_hf), ("pred", rho_pred)]:
            k, Pk = power_spectrum(F, self.box_size, self.n_bins)
            ax.loglog(k[1:], Pk[1:], label=label, lw=1.7)
            out[f"P_{label}"] = {"k": k.tolist(), "P": Pk.tolist()}
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$P_\delta(k)$")
        ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_density_pk.png", dpi=120)
        plt.close(fig)

        # T(k) and r(k)
        fig, ax = plt.subplots(figsize=(7, 5))
        k, T, _, _ = transfer_function(rho_pred, rho_hf, self.box_size, self.n_bins)
        _, rk = coherence(rho_pred, rho_hf, self.box_size, self.n_bins)
        ax.semilogx(k[1:], T[1:],  label="T(k)", lw=1.7)
        ax.semilogx(k[1:], rk[1:], label="r(k)", lw=1.7)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$k$"); ax.set_ylim(-0.05, 1.4)
        ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_density_T_r.png", dpi=120)
        plt.close(fig)
        out["T"] = T.tolist(); out["r"] = rk.tolist(); out["k"] = k.tolist()

        # 2D mean slice
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax_, F, title in zip(axes, [rho_lf, rho_hf, rho_pred],
                                 ["LF", "HF (truth)", "HF (pred)"]):
            img = F.mean(axis=0)
            vmax = float(np.percentile(np.abs(img), 99))
            ax_.imshow(img, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax_.set_title(f"{title} density"); ax_.axis("off")
        fig.tight_layout()
        fig.savefig(self.out_dir / f"set{r.set_id}_density_slice.png", dpi=120)
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def run_one(self, set_id: int, extent: tuple[int, int, int],
                save_arrays: bool = False) -> dict:
        r = self.predict_full_volume(set_id, extent)
        stats: dict = {"set_id": set_id, "extent": list(extent),
                       "box_size": self.box_size,
                       "crop_size": self.D, "crop_overlap": self.overlap,
                       "steps": self.steps}
        self._slices_plot(r)
        stats["disp_pk"] = {f"disp{c}": self._pk_plot(r, c) for c in range(3)}
        stats["disp_T"]  = self._T_plot(r)
        stats["disp_r"]  = self._r_plot(r)
        stats["disp_residual"] = self._residual_hist(r)
        stats["density"] = self._density_panels(r)
        with open(self.out_dir / f"set{set_id}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        if save_arrays:
            np.savez_compressed(self.out_dir / f"set{set_id}_cubes.npz",
                                lf=r.lf, hf=r.hf, pred=r.pred)
        print(f"[eval] set {set_id} → {self.out_dir}")
        return stats

    def run(self,
            test_sets: Optional[Iterable[tuple[int, tuple[int, int, int]]]] = None,
            max_sets: Optional[int] = None,
            save_arrays: bool = False) -> list[dict]:
        if test_sets is None:
            sets = discover_sets(self.cfg["data"]["root"], self.reader,
                                 self.snapshot)
            test_sets = split_sets(sets)["test"]
        test_sets = list(test_sets)
        if max_sets is not None:
            test_sets = test_sets[:max_sets]
        print(f"[eval] running {len(test_sets)} test sets: {test_sets}")
        return [self.run_one(sid, ext, save_arrays=save_arrays)
                for sid, ext in test_sets]
