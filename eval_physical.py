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
from ops.geometry import overlap_crop_starts
from ops.spectrum import (amplitude_match, coherence, power_spectrum,
                          rescale_by_curve, transfer_function)


def _axis_split_bounds(starts, D: int, L_axis: int) -> list[int]:
    """Per-axis voxel-ownership boundaries between adjacent crops.

    Given crop starts s_0 < s_1 < ... < s_{K-1} (each crop has length D),
    returns b of length K+1 such that crop k owns voxel indices
    [b[k], b[k+1]) along this axis. Adjacent crops split at the integer
    midpoint of their overlap region — for uniform stride D-d, this places
    the cut at s_k + D - d/2, matching the training-time inner cube.

    Inlined from inference_distributed.axis_split_bounds to avoid pulling
    the torch.distributed stack into a single-process eval script.
    """
    K = len(starts)
    if K == 0:
        raise ValueError("starts must be non-empty")
    if K == 1:
        return [0, L_axis]
    bounds = [0]
    for k in range(K - 1):
        s_next = starts[k + 1]
        e_curr = starts[k] + D
        if s_next >= e_curr:
            raise ValueError(
                f"adjacent crops do not overlap: end[{k}]={e_curr} <= "
                f"start[{k+1}]={s_next}"
            )
        bounds.append((s_next + e_curr) // 2)
    bounds.append(L_axis)
    return bounds


def _load_model_and_norms(ckpt_path: str, device: str, use_ema: bool):
    # Load on CPU then move the model to the device. Deserializing tensors
    # straight onto CUDA (map_location=device) can raise "CUDA device busy/
    # unavailable" on a contended node; loading on CPU avoids that path.
    payload = CheckpointManager.load(ckpt_path, map_location="cpu")
    cfg = payload["cfg"]
    norm = NormStats.from_dict(payload["norm"])
    extra_norms = {k: NormStats.from_dict(v)
                   for k, v in (payload.get("extra_norms") or {}).items()}

    model = PVFlowMatcher.from_config(cfg).to(device)

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
    filled    = np.zeros((Lx, Ly, Lz), dtype=bool)

    D = ds.D
    d = ds.overlap
    norm = ds.norm

    # Per-axis ownership: each crop owns disjoint inner block [b[k], b[k+1])
    # along each axis, splitting overlaps at the integer midpoint. The union
    # of owned blocks tiles the volume exactly — no averaging, no buffer
    # contamination. Matches inference_distributed.py and the spec.
    starts_x = overlap_crop_starts(Lx, D, d)
    starts_y = overlap_crop_starts(Ly, D, d)
    starts_z = overlap_crop_starts(Lz, D, d)
    bounds_x = _axis_split_bounds(starts_x, D, Lx)
    bounds_y = _axis_split_bounds(starts_y, D, Ly)
    bounds_z = _axis_split_bounds(starts_z, D, Lz)
    sx_to_k = {sx: k for k, sx in enumerate(starts_x)}
    sy_to_k = {sy: k for k, sy in enumerate(starts_y)}
    sz_to_k = {sz: k for k, sz in enumerate(starts_z)}

    for ci in crop_indices:
        crop = ds[ci]
        # Voxel-domain LF and HF cubes (denormalized)
        lf_vox = norm.denormalize(crop["lf_voxel"][:3].numpy())   # (3,D,D,D)
        tgt_vox = crop["tgt_vox"].numpy()                          # (3,D,D,D)
        residual_phys = norm.denormalize_residual(tgt_vox)
        hf_vox = lf_vox + residual_phys

        batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v)
                 for k, v in crop.items()}
        pred_pt = _predict_residual(
            model, batch, device, mode, steps).cpu().numpy()[0]   # (D^3, 3)
        pred_vox = np.zeros_like(tgt_vox)
        ix = ds._cell_idx[:, 0]; iy = ds._cell_idx[:, 1]; iz = ds._cell_idx[:, 2]
        for c in range(3):
            pred_vox[c, ix, iy, iz] = pred_pt[:, c]
        pred_vox_phys = norm.denormalize_residual(pred_vox)
        pred_hf_vox = lf_vox + pred_vox_phys

        sx, sy, sz = ds.crops[ci][1:4]
        kx, ky, kz = sx_to_k[sx], sy_to_k[sy], sz_to_k[sz]
        # Owned global voxel range for this crop
        gx0, gx1 = bounds_x[kx], bounds_x[kx + 1]
        gy0, gy1 = bounds_y[ky], bounds_y[ky + 1]
        gz0, gz1 = bounds_z[kz], bounds_z[kz + 1]
        # Corresponding local slice into this crop's (D, D, D) cube
        lx0, lx1 = gx0 - sx, gx1 - sx
        ly0, ly1 = gy0 - sy, gy1 - sy
        lz0, lz1 = gz0 - sz, gz1 - sz

        lf_full[:, gx0:gx1, gy0:gy1, gz0:gz1]   = lf_vox[:, lx0:lx1, ly0:ly1, lz0:lz1]
        hf_full[:, gx0:gx1, gy0:gy1, gz0:gz1]   = hf_vox[:, lx0:lx1, ly0:ly1, lz0:lz1]
        pred_full[:, gx0:gx1, gy0:gy1, gz0:gz1] = pred_hf_vox[:, lx0:lx1, ly0:ly1, lz0:lz1]
        filled[gx0:gx1, gy0:gy1, gz0:gz1] = True

    n_unfilled = int((~filled).sum())
    if n_unfilled:
        print(f"[physical] warning: {n_unfilled} voxels left unfilled in set {sid_filter}")
    return (lf_full, hf_full, pred_full, ext_vox)


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
    p.add_argument("--root", type=str, default=None,
                   help="override data.root from the checkpoint cfg. Needed "
                        "when training staged data to a node-local path "
                        "(e.g. /scratch/...) that does not exist at eval time.")
    p.add_argument("--spectral_correct", action="store_true",
                   help="also report an ORACLE phase-preserving amplitude "
                        "correction (per-shell rescale to HF power): T->1, r "
                        "preserved. Upper bound for a spectral-amplitude method.")
    p.add_argument("--split", type=str, default="val",
                   choices=["train", "val", "test"],
                   help="which split to evaluate (default val). Use 'train' "
                        "with --dump_tk to build a deployable correction curve.")
    p.add_argument("--dump_tk", type=str, default=None,
                   help="save the mean model->HF transfer curve (k, T) over the "
                        "evaluated sets to this .npz (build the deployable curve "
                        "from the TRAIN split).")
    p.add_argument("--correct_curve", type=str, default=None,
                   help="apply a DEPLOYABLE (non-oracle) amplitude correction "
                        "using a pre-measured T(k) curve .npz (from --dump_tk on "
                        "train); reports deploy-corrected T/r. r is preserved.")
    p.add_argument("--sids", type=str, default=None,
                   help="comma-separated set ids to evaluate (overrides the "
                        "first-max_sets selection). Use to match extent/box "
                        "between the curve-building and correction sets.")
    p.add_argument("--dump_density", type=str, default=None,
                   help="directory to save per-set reconstructed density fields "
                        "(rho_hf, rho_lf, rho_pred, style) for held-out "
                        "statistics (bispectrum) and posterior P(k) summaries.")
    p.add_argument("--dump_tk_perset", type=str, default=None,
                   help="save per-set (sid, style, k, T) to .npz for fitting a "
                        "cosmology-conditioned transfer emulator.")
    p.add_argument("--correct_emulator", type=str, default=None,
                   help="apply a DEPLOYABLE cosmology-conditioned amplitude "
                        "correction: emulator .npz (k, coef) predicts T(k) from "
                        "the set's 5-param cosmology; r preserved. Non-oracle.")
    args = p.parse_args()

    model, cfg, norm, extra_norms, epoch = _load_model_and_norms(
        args.ckpt, args.device, args.use_ema)
    if args.root:
        cfg["data"]["root"] = args.root
        print(f"[physical] data.root overridden -> {args.root}")
    mode = (cfg.get("flow") or {}).get("mode", "flow_matching")
    box_size = cfg["data"].get("box_size", 1000.0)

    print(f"[physical] checkpoint epoch {epoch}  mode={mode}  "
          f"use_ema={args.use_ema}  box={box_size} Mpc/h")

    reader = get_reader(cfg["data"].get("reader", "numpy"))
    datasets, _, _ = build_datasets(cfg, reader=reader, norm=norm,
                                    extra_norms=extra_norms or None)
    val_ds = datasets[args.split]
    all_sids = sorted({sid for sid, *_ in val_ds.crops})
    if args.sids:
        want = [int(s) for s in args.sids.split(",")]
        sids = [s for s in want if s in all_sids][:args.max_sets]
        missing = [s for s in want if s not in all_sids]
        if missing:
            print(f"[physical] warning: sids not in {args.split} split: {missing}")
    else:
        sids = all_sids[:args.max_sets]
    print(f"[physical] evaluating split={args.split} sets: {sids}")

    # Deployable correction curve (from --dump_tk on train); loaded once.
    curve = None
    if args.correct_curve:
        cz = np.load(args.correct_curve)
        curve = (cz["k"], cz["T"])
        print(f"[physical] deployable correction from {args.correct_curve} "
              f"({len(cz['k'])} k-bins) — NON-oracle, r-preserving")
    emu = None
    if args.correct_emulator:
        ez = np.load(args.correct_emulator)
        emu = (ez["k"], ez["coef"])   # coef: (n_bins, n_feat), feat=[1, style(5)]
        print(f"[physical] cosmology-conditioned emulator from "
              f"{args.correct_emulator} — NON-oracle, r-preserving")
    tk_k_accum, tk_T_accum = [], []          # for --dump_tk (mean curve)
    ps_sid, ps_style, ps_k, ps_T = [], [], [], []   # for --dump_tk_perset

    root = cfg["data"]["root"]
    snap = cfg["data"].get("snapshot", "PART_009")

    def _load_style(sid: int) -> np.ndarray:
        import os
        p = os.path.join(root, "quijote-64", f"set{sid}_pos_0_0_0",
                         snap, "style.npy")
        return np.load(p).astype(np.float64)

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

        # Optionally save the density fields for held-out statistics
        # (bispectrum, wavelet) and for the posterior P(k) summaries.
        if args.dump_density:
            ddir = Path(args.dump_density)
            ddir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ddir / f"set{sid}_density.npz",
                                rho_hf=rho_hf.astype(np.float32),
                                rho_lf=rho_lf.astype(np.float32),
                                rho_pred=rho_pred.astype(np.float32),
                                box_size=float(box_size), L=int(L),
                                style=_load_style(sid))

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

        row = {
            "set":          sid,
            "T_pred_err":   T_pred_err,
            "T_lf_err":     T_lf_err,
            "r_pred":       r_pred_avg,
            "r_lf":         r_lf_avg,
            "L":            L,
        }
        msg = (f"  set {sid}: |T-1| pred={T_pred_err:.3f}  lf={T_lf_err:.3f}   "
               f"r pred={r_pred_avg:.3f}  lf={r_lf_avg:.3f}")

        # Oracle phase-preserving amplitude correction (upper bound): rescale
        # the predicted density's per-shell Fourier amplitude to match HF power.
        # r(k) is invariant under this, T(k)->1 => isolates amplitude vs phase.
        if args.spectral_correct:
            rho_pred_corr = amplitude_match(rho_pred, rho_hf, box_size=box_size)
            _, T_c, _, _ = transfer_function(rho_pred_corr, rho_hf,
                                             box_size=box_size, n_bins=args.n_kbins)
            _, r_c = coherence(rho_pred_corr, rho_hf, box_size=box_size,
                               n_bins=args.n_kbins)
            row["T_corr_err"] = float(np.mean(np.abs(T_c[lo:hi] - 1.0)))
            row["r_corr"]     = float(np.mean(r_c[lo:hi]))
            msg += (f"  |  ORACLE |T-1|={row['T_corr_err']:.3f} r={row['r_corr']:.3f}")

        # Deployable (non-oracle) correction: rescale by a fixed T(k) curve
        # measured on the TRAIN split — carries no info from this val cube.
        if curve is not None:
            rho_dep = rescale_by_curve(rho_pred, curve[0], curve[1],
                                       box_size=box_size)
            _, T_d, _, _ = transfer_function(rho_dep, rho_hf,
                                             box_size=box_size, n_bins=args.n_kbins)
            _, r_d = coherence(rho_dep, rho_hf, box_size=box_size,
                               n_bins=args.n_kbins)
            row["T_dep_err"] = float(np.mean(np.abs(T_d[lo:hi] - 1.0)))
            row["r_dep"]     = float(np.mean(r_d[lo:hi]))
            msg += (f"  |  DEPLOY |T-1|={row['T_dep_err']:.3f} r={row['r_dep']:.3f}")

        # Deployable cosmology-conditioned correction: predict T(k) from the
        # set's 5-param cosmology via the fitted emulator, then rescale.
        if emu is not None:
            style_vec = _load_style(sid)
            feat = np.concatenate([[1.0], style_vec])            # (1+5,)
            T_hat = emu[1] @ feat                                # (n_bins,)
            rho_emu = rescale_by_curve(rho_pred, emu[0], T_hat, box_size=box_size)
            _, T_e, _, _ = transfer_function(rho_emu, rho_hf,
                                             box_size=box_size, n_bins=args.n_kbins)
            _, r_e = coherence(rho_emu, rho_hf, box_size=box_size,
                               n_bins=args.n_kbins)
            row["T_emu_err"] = float(np.mean(np.abs(T_e[lo:hi] - 1.0)))
            row["r_emu"]     = float(np.mean(r_e[lo:hi]))
            msg += (f"  |  EMU |T-1|={row['T_emu_err']:.3f} r={row['r_emu']:.3f}")

        if args.dump_tk is not None:
            tk_k_accum.append(k)
            tk_T_accum.append(T_pred)
        if args.dump_tk_perset is not None:
            ps_sid.append(sid); ps_style.append(_load_style(sid))
            ps_k.append(k); ps_T.append(T_pred)

        summary_rows.append(row)
        print(msg)

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
        if args.spectral_correct and "T_corr_err" in summary_rows[0]:
            T_c_avg = float(np.mean([r["T_corr_err"] for r in summary_rows]))
            r_c_avg = float(np.mean([r["r_corr"]     for r in summary_rows]))
            print(f"   [ORACLE amplitude-corrected] |T-1|={T_c_avg:.3f}  "
                  f"r={r_c_avg:.3f}  (phase-preserving upper bound)")
        if curve is not None and "T_dep_err" in summary_rows[0]:
            T_d_avg = float(np.mean([r["T_dep_err"] for r in summary_rows]))
            r_d_avg = float(np.mean([r["r_dep"]     for r in summary_rows]))
            print(f"   [DEPLOY curve-corrected]    |T-1|={T_d_avg:.3f}  "
                  f"r={r_d_avg:.3f}  (train-derived mean curve, non-oracle)")
        if emu is not None and "T_emu_err" in summary_rows[0]:
            T_e_avg = float(np.mean([r["T_emu_err"] for r in summary_rows]))
            r_e_avg = float(np.mean([r["r_emu"]     for r in summary_rows]))
            print(f"   [DEPLOY emulator-corrected] |T-1|={T_e_avg:.3f}  "
                  f"r={r_e_avg:.3f}  (cosmology-conditioned T(k), non-oracle)")

    # Save the deployable transfer curve (mean over evaluated sets).
    if args.dump_tk is not None and tk_T_accum:
        k_ref = tk_k_accum[0]
        T_stack = np.stack([np.interp(k_ref, kk, TT)
                            for kk, TT in zip(tk_k_accum, tk_T_accum)])
        np.savez(args.dump_tk, k=k_ref, T=T_stack.mean(axis=0))
        print(f"[physical] wrote transfer curve ({len(k_ref)} bins, "
              f"mean over {len(tk_T_accum)} sets) -> {args.dump_tk}")

    # Save per-set (style, T) for fitting a cosmology-conditioned emulator.
    if args.dump_tk_perset is not None and ps_T:
        k_ref = ps_k[0]
        T_stack = np.stack([np.interp(k_ref, kk, TT)
                            for kk, TT in zip(ps_k, ps_T)])
        np.savez(args.dump_tk_perset, sid=np.array(ps_sid),
                 style=np.stack(ps_style), k=k_ref, T=T_stack)
        print(f"[physical] wrote per-set transfer data "
              f"({len(ps_T)} sets, {len(k_ref)} bins) -> {args.dump_tk_perset}")

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
