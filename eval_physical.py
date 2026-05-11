"""Physical / cosmological evaluation: transfer function + coherence.

For a trained checkpoint, generates predicted residuals on the validation
set, reconstructs HF displacement (= LF + residual), converts to density
via CIC deposition, then computes:

  - P_HF(k):     true HF density power spectrum
  - P_pred(k):   predicted HF density power spectrum
  - P_LF(k):     LF (no-correction) density power spectrum
  - T(k):        sqrt(P_pred / P_HF)   -- amplitude transfer
  - r(k):        cross-coherence       -- phase agreement
  - T_LF(k), r_LF(k): same for LF baseline (predict-zero residual)

Reports per-bin numbers plus a few summary scalars.

Usage:
    python eval_physical.py --ckpt runs/pvfm_*/ckpt_latest.pt
    python eval_physical.py --ckpt ... --max_sets 4 --use_ema --steps 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data import (NormStats, build_dataloaders, build_datasets, get_reader,
                  SimulationDataset)
from engine import (CheckpointManager, direct_sample, euler_sample,
                    lf_init_sample)
from models import PVFlowMatcher
from ops.density import disp_to_density
from ops.spectrum import coherence, power_spectrum, transfer_function


def _load_model_and_norms(ckpt_path: str, device: str, use_ema: bool):
    payload = CheckpointManager.load(ckpt_path, map_location=device)
    cfg = payload["cfg"]
    norm = NormStats.from_dict(payload["norm"])
    extra_norms = {k: NormStats.from_dict(v)
                   for k, v in (payload.get("extra_norms") or {}).items()}

    m = cfg["model"]
    default_c_env = 4 if cfg["data"].get("env_outside_mask", True) else 3
    n_fields = len(cfg["data"].get("fields", ["disp"]))
    model = PVFlowMatcher(
        c_pt=3, c_lf=m.get("c_lf", 3 * n_fields),
        c_env=m.get("c_env", default_c_env),
        c_lf_pt=m.get("c_lf_pt", 3 * n_fields),
        n_style=m.get("n_style", 5),
        base_voxel=m.get("base_voxel", 32),
        base_point=m.get("base_point", 128),
        cond_dim=m.get("cond_dim", 256),
        n_blocks=m.get("n_blocks", 4),
        env_resolution=m.get("env_resolution", 64),
    ).to(device)

    state = (payload["ema"] if (use_ema and payload.get("ema"))
             else payload["model"])
    model.load_state_dict(state)
    model.eval()
    return model, cfg, norm, extra_norms, payload["epoch"]


@torch.no_grad()
def _predict_residual(model, batch, device, mode: str, steps: int):
    lf_voxel = batch["lf_voxel"].to(device)
    env      = batch["env"].to(device)
    coords   = batch["coords"].to(device)
    lf_pt    = batch["lf_pt"].to(device)
    style    = batch["style"].to(device)
    if mode == "direct":
        return direct_sample(model, lf_voxel, env, style, coords, lf_pt)
    if mode == "lf_init":
        return lf_init_sample(model, lf_voxel, env, style, coords, lf_pt,
                              steps=steps)
    return euler_sample(model, lf_voxel, env, style, coords, lf_pt, steps=steps)


def _reconstruct_cube(ds: SimulationDataset, model, device, mode: str,
                      steps: int, sid_filter: int) -> tuple:
    """Stitch predicted+true displacement cubes for one full simulation set.

    Returns (lf_disp, hf_disp, pred_disp) each shape (3, L, L, L) in
    physical (denormalized) units — ready for disp_to_density.
    """
    # Identify all crops for this sid and the simulation extent.
    crop_indices = [i for i, (sid, *_rest) in enumerate(ds.crops)
                    if sid == sid_filter]
    if not crop_indices:
        raise ValueError(f"set_id {sid_filter} not present in dataset")
    sid, _, _, _, ext_vox = ds.crops[crop_indices[0]]
    Lx, Ly, Lz = ext_vox

    lf_full   = np.zeros((3, Lx, Ly, Lz), dtype=np.float32)
    hf_full   = np.zeros((3, Lx, Ly, Lz), dtype=np.float32)
    pred_full = np.zeros((3, Lx, Ly, Lz), dtype=np.float32)
    weight    = np.zeros((Lx, Ly, Lz), dtype=np.float32)

    D = ds.D
    norm = ds.norm

    for ci in crop_indices:
        crop = ds[ci]
        # Voxel-domain LF and HF cubes (denormalized)
        lf_vox = norm.denormalize(crop["lf_voxel"][:3].numpy())   # (3,D,D,D)
        tgt_vox = crop["tgt_vox"].numpy()                          # (3,D,D,D)
        std = norm.std.reshape(3, 1, 1, 1)
        residual_phys = tgt_vox * std
        hf_vox = lf_vox + residual_phys

        batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v)
                 for k, v in crop.items()}
        pred_pt = _predict_residual(
            model, batch, device, mode, steps).cpu().numpy()[0]   # (D^3, 3)
        pred_vox = np.zeros_like(tgt_vox)
        ix = ds._cell_idx[:, 0]; iy = ds._cell_idx[:, 1]; iz = ds._cell_idx[:, 2]
        for c in range(3):
            pred_vox[c, ix, iy, iz] = pred_pt[:, c]
        pred_vox_phys = pred_vox * std
        pred_hf_vox = lf_vox + pred_vox_phys

        sx, sy, sz = ds.crops[ci][1:4]
        # Cover the entire crop region (no inner-only mask). Overlap is
        # handled by uniform averaging (weight += 1 per contributing crop).
        gx0, gx1 = sx, min(sx + D, Lx)
        gy0, gy1 = sy, min(sy + D, Ly)
        gz0, gz1 = sz, min(sz + D, Lz)
        lx, ly, lz = gx1 - gx0, gy1 - gy0, gz1 - gz0
        lf_full[:, gx0:gx1, gy0:gy1, gz0:gz1]   += lf_vox[:, :lx, :ly, :lz]
        hf_full[:, gx0:gx1, gy0:gy1, gz0:gz1]   += hf_vox[:, :lx, :ly, :lz]
        pred_full[:, gx0:gx1, gy0:gy1, gz0:gz1] += pred_hf_vox[:, :lx, :ly, :lz]
        weight[gx0:gx1, gy0:gy1, gz0:gz1]       += 1.0

    n_unfilled = int((weight == 0).sum())
    if n_unfilled:
        print(f"[physical] warning: {n_unfilled} voxels left unfilled in set {sid_filter}")
    safe = np.maximum(weight[None], 1.0)
    return (lf_full / safe, hf_full / safe, pred_full / safe, ext_vox)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--max_sets", type=int, default=2,
                   help="number of full simulations to evaluate (default 2)")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_kbins", type=int, default=24)
    args = p.parse_args()

    model, cfg, norm, extra_norms, epoch = _load_model_and_norms(
        args.ckpt, args.device, args.use_ema)
    mode = (cfg.get("flow") or {}).get("mode", "flow_matching")
    box_size = cfg["data"].get("box_size", 1000.0)

    print(f"[physical] checkpoint epoch {epoch}  mode={mode}  "
          f"use_ema={args.use_ema}  box={box_size} Mpc/h")

    reader = get_reader(cfg["data"].get("reader", "numpy"))
    datasets, _, _ = build_datasets(cfg, reader=reader, norm=norm,
                                    extra_norms=extra_norms or None)
    val_ds = datasets["val"]
    sids = sorted({sid for sid, *_ in val_ds.crops})[:args.max_sets]
    print(f"[physical] evaluating sets: {sids}")

    out_dir = Path(args.out_dir or (Path(args.ckpt).parent / "physical"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for sid in sids:
        print(f"[physical] reconstructing set {sid} ...")
        lf, hf, pred, ext = _reconstruct_cube(
            val_ds, model, args.device, mode, args.steps, sid)
        L = ext[0]
        # Use only the first (smallest) component cube — Quijote sets are
        # always cubic L×L×L per the dataset.
        # Compute density from each disp field. Box length in voxels.
        rho_lf   = disp_to_density(lf,   box_size=box_size)
        rho_hf   = disp_to_density(hf,   box_size=box_size)
        rho_pred = disp_to_density(pred, box_size=box_size)

        k, T_pred, P_pred, P_hf = transfer_function(
            rho_pred, rho_hf, box_size=box_size, n_bins=args.n_kbins)
        _, r_pred = coherence(rho_pred, rho_hf, box_size=box_size,
                              n_bins=args.n_kbins)
        _, T_lf, P_lf, _ = transfer_function(
            rho_lf, rho_hf, box_size=box_size, n_bins=args.n_kbins)
        _, r_lf = coherence(rho_lf, rho_hf, box_size=box_size,
                            n_bins=args.n_kbins)

        # Per-set CSV
        with open(out_dir / f"set{sid}_spectra.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "P_hf", "P_lf", "P_pred",
                        "T_lf", "T_pred", "r_lf", "r_pred"])
            for i in range(len(k)):
                w.writerow([k[i], P_hf[i], P_lf[i], P_pred[i],
                            T_lf[i], T_pred[i], r_lf[i], r_pred[i]])

        # Summary metrics: mean abs(T - 1) over a "useful" k range
        # and mean coherence over the same range. Use middle 60% of bins
        # to ignore noisy lowest k (large modes) and Nyquist edge.
        lo = max(1, len(k) // 5)
        hi = max(lo + 1, 4 * len(k) // 5)
        T_pred_err = float(np.mean(np.abs(T_pred[lo:hi] - 1.0)))
        T_lf_err   = float(np.mean(np.abs(T_lf[lo:hi]   - 1.0)))
        r_pred_avg = float(np.mean(r_pred[lo:hi]))
        r_lf_avg   = float(np.mean(r_lf[lo:hi]))

        summary_rows.append({
            "set":          sid,
            "T_pred_err":   T_pred_err,
            "T_lf_err":     T_lf_err,
            "r_pred":       r_pred_avg,
            "r_lf":         r_lf_avg,
            "L":            L,
        })
        print(f"  set {sid}: |T-1| pred={T_pred_err:.3f}  lf={T_lf_err:.3f}   "
              f"r pred={r_pred_avg:.3f}  lf={r_lf_avg:.3f}")

    # Overall summary
    if summary_rows:
        T_pred_avg = float(np.mean([r["T_pred_err"] for r in summary_rows]))
        T_lf_avg   = float(np.mean([r["T_lf_err"]   for r in summary_rows]))
        r_p        = float(np.mean([r["r_pred"]     for r in summary_rows]))
        r_l        = float(np.mean([r["r_lf"]       for r in summary_rows]))

        print()
        print(f"[physical] overall (k middle 60%):")
        print(f"   |T-1| pred={T_pred_avg:.3f}  lf={T_lf_avg:.3f}  "
              f"(lower better; 0 = perfect amplitude)")
        print(f"   r     pred={r_p:.3f}        lf={r_l:.3f}        "
              f"(higher better; 1 = perfect phase)")

        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set", "T_pred_err", "T_lf_err", "r_pred", "r_lf", "L"])
            for r in summary_rows:
                w.writerow([r["set"], f"{r['T_pred_err']:.6e}",
                            f"{r['T_lf_err']:.6e}", f"{r['r_pred']:.6e}",
                            f"{r['r_lf']:.6e}", r["L"]])
        print(f"[physical] wrote per-set spectra and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
