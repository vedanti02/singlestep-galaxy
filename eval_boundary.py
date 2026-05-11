"""Boundary-error diagnostic for the region-based single-step model.

Computes per-point squared prediction error vs. L_inf distance to the
nearest crop face, aggregated over the validation split. Reads the curve
to decide whether the loss-mask buffer width (``crop_overlap // 2``) is
correctly sized.

Usage:
    python eval_boundary.py --ckpt runs/pvfm/ckpt_latest.pt
    python eval_boundary.py --ckpt ... --max_batches 8 --use_ema
    python eval_boundary.py --ckpt ... --steps 4 --out_dir runs/pvfm/boundary_step4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data import (NormStats, build_dataloaders, build_datasets, get_reader)
from engine import (CheckpointManager, direct_sample, euler_sample,
                    lf_init_sample)
from models import PVFlowMatcher


def build_distance_lut(D: int) -> np.ndarray:
    """L_inf distance from each cell to the nearest face, flattened to (D**3,)."""
    ii, jj, kk = np.meshgrid(np.arange(D), np.arange(D), np.arange(D),
                             indexing="ij")
    dx = np.minimum(ii, D - 1 - ii)
    dy = np.minimum(jj, D - 1 - jj)
    dz = np.minimum(kk, D - 1 - kk)
    return np.minimum(np.minimum(dx, dy), dz).ravel().astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: <ckpt-parent>/boundary")
    p.add_argument("--max_batches", type=int, default=None,
                   help="cap number of validation batches (smoke-test mode)")
    p.add_argument("--steps", type=int, default=1,
                   help="Euler integration steps (default 1 = single-step)")
    p.add_argument("--n_ensemble", type=int, default=1,
                   help="average N noise samples per crop (FM only). N>1 cuts "
                        "the 1-step noise floor by 1/sqrt(N).")
    p.add_argument("--force_init", choices=["noise", "lf"], default=None,
                   help="override the inference init for FM-trained models. "
                        "'lf' uses lf_init_sample (start at LF, no noise floor) "
                        "even if cfg.flow.mode is the default flow_matching.")
    p.add_argument("--use_ema", action="store_true",
                   help="evaluate the EMA shadow weights if the ckpt has them")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="torch RNG seed for the noise in euler_sample")
    args = p.parse_args()

    payload = CheckpointManager.load(args.ckpt, map_location=args.device)
    cfg = payload["cfg"]
    norm = NormStats.from_dict(payload["norm"])
    extra_norms_raw = payload.get("extra_norms") or {}
    extra_norms = {k: NormStats.from_dict(v) for k, v in extra_norms_raw.items()}

    m = cfg["model"]
    default_c_env = 4 if cfg["data"].get("env_outside_mask", True) else 3
    n_fields = len(cfg["data"].get("fields", ["disp"]))
    c_lf = m.get("c_lf", 3 * n_fields)
    c_lf_pt = m.get("c_lf_pt", 3 * n_fields)
    model = PVFlowMatcher(
        c_pt=3, c_lf=c_lf, c_env=m.get("c_env", default_c_env), c_lf_pt=c_lf_pt,
        n_style=m.get("n_style", 5),
        base_voxel=m.get("base_voxel", 32),
        base_point=m.get("base_point", 128),
        cond_dim=m.get("cond_dim", 256),
        n_blocks=m.get("n_blocks", 4),
        env_resolution=m.get("env_resolution", 64),
    ).to(args.device)

    state = (payload["ema"] if (args.use_ema and payload.get("ema"))
             else payload["model"])
    model.load_state_dict(state)
    model.eval()
    print(f"[boundary] loaded {'EMA' if args.use_ema else 'live'} weights "
          f"from {args.ckpt} (epoch {payload['epoch']})")

    reader = get_reader(cfg["data"].get("reader", "numpy"))
    datasets, _, _ = build_datasets(cfg, reader=reader, norm=norm,
                                    extra_norms=extra_norms or None)
    if "val" not in datasets:
        raise SystemExit("[boundary] no validation split available")
    loaders = build_dataloaders(cfg, {"val": datasets["val"]})
    val_loader = loaders["val"]

    D = cfg["data"]["crop_size"]
    overlap = cfg["data"]["crop_overlap"]
    buf = overlap // 2
    dist_lut = build_distance_lut(D)
    n_bins = int(dist_lut.max()) + 1
    per_crop_count = np.bincount(dist_lut, minlength=n_bins).astype(np.int64)

    sum_sq      = np.zeros(n_bins, dtype=np.float64)   # ||x1_hat - tgt||^2
    sum_sq_zero = np.zeros(n_bins, dtype=np.float64)   # ||0       - tgt||^2 = ||tgt||^2
    sum_sq_lf   = np.zeros(n_bins, dtype=np.float64)   # ||lf_pt   - tgt||^2  (predict LF copy)
    n_crops = 0

    torch.manual_seed(args.seed)
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            lf_voxel = batch["lf_voxel"].to(args.device)
            env      = batch["env"].to(args.device)
            coords   = batch["coords"].to(args.device)
            lf_pt    = batch["lf_pt"].to(args.device)
            tgt_pt   = batch["tgt_pt"].to(args.device)
            style    = batch["style"].to(args.device)

            mode = cfg["flow"].get("mode", "flow_matching") if cfg.get("flow") else "flow_matching"
            if args.force_init == "lf":
                x1_hat = lf_init_sample(model, lf_voxel, env, style,
                                        coords, lf_pt, steps=args.steps)
            elif mode == "direct":
                x1_hat = direct_sample(model, lf_voxel, env, style,
                                       coords, lf_pt)
            elif mode == "lf_init":
                x1_hat = lf_init_sample(model, lf_voxel, env, style,
                                        coords, lf_pt, steps=args.steps)
            elif args.n_ensemble > 1:
                acc = None
                for _ in range(args.n_ensemble):
                    one = euler_sample(model, lf_voxel, env, style,
                                       coords, lf_pt, steps=args.steps)
                    acc = one if acc is None else acc + one
                x1_hat = acc / args.n_ensemble
            else:
                x1_hat = euler_sample(model, lf_voxel, env, style,
                                      coords, lf_pt, steps=args.steps)

            # Per-point squared L2 errors. tgt_pt is the HF-LF residual in
            # normalized space; the model predicts that same residual.
            # Baselines:
            #   - "zero": predict zero residual (i.e. HF == LF). MSE here
            #     is the ||tgt||^2 norm itself, so it doubles as the
            #     per-bin variance proxy of the target.
            #   - "lf":   predict the LF displacement at the point as the
            #     residual. Sanity: should be much worse than zero, since
            #     residuals are tiny but LF disp is unit-scale.
            err      = ((x1_hat - tgt_pt) ** 2).sum(dim=-1)
            err_zero = (tgt_pt ** 2).sum(dim=-1)
            # Only the first 3 channels of lf_pt are disp; vel channels
            # (if present) are not comparable to the disp residual target.
            err_lf   = ((lf_pt[..., :3] - tgt_pt) ** 2).sum(dim=-1)
            err_np      = err.detach().cpu().numpy().astype(np.float64)
            err_zero_np = err_zero.detach().cpu().numpy().astype(np.float64)
            err_lf_np   = err_lf.detach().cpu().numpy().astype(np.float64)

            for b in range(err_np.shape[0]):
                sum_sq      += np.bincount(dist_lut, weights=err_np[b],
                                           minlength=n_bins)
                sum_sq_zero += np.bincount(dist_lut, weights=err_zero_np[b],
                                           minlength=n_bins)
                sum_sq_lf   += np.bincount(dist_lut, weights=err_lf_np[b],
                                           minlength=n_bins)
            n_crops += err_np.shape[0]
            print(f"[boundary] batch {bi + 1}  crops_seen={n_crops}")

    if n_crops == 0:
        raise SystemExit("[boundary] no batches processed; check val split")

    count = n_crops * per_crop_count
    safe = np.maximum(count, 1)
    mean_sq      = np.where(count > 0, sum_sq      / safe, np.nan)
    mean_sq_zero = np.where(count > 0, sum_sq_zero / safe, np.nan)
    mean_sq_lf   = np.where(count > 0, sum_sq_lf   / safe, np.nan)
    # Fraction-of-target-variance error: < 1 means model beats predict-zero,
    # > 1 means model is worse than just predicting HF == LF.
    rel_to_zero  = np.where(mean_sq_zero > 0, mean_sq / mean_sq_zero, np.nan)

    out_dir = Path(args.out_dir or (Path(args.ckpt).parent / "boundary"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "boundary_error.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_from_edge", "mean_sq_error",
                    "baseline_zero", "baseline_lf", "rel_to_zero",
                    "count"])
        for d in range(n_bins):
            w.writerow([d,
                        f"{mean_sq[d]:.6e}",
                        f"{mean_sq_zero[d]:.6e}",
                        f"{mean_sq_lf[d]:.6e}",
                        f"{rel_to_zero[d]:.4f}",
                        int(count[d])])

    print()
    print(f"[boundary] crop_size D={D}, crop_overlap d={overlap}, "
          f"buf=d/2={buf}  (inner cube: dist >= {buf})")
    print(f"[boundary] crops processed: {n_crops}")
    print(f"[boundary] wrote {csv_path}")
    print()
    print(f"{'dist':>4} {'model_mse':>12} {'zero_mse':>12} "
          f"{'lf_mse':>12} {'rel_zero':>9} {'count':>10} {'region':>8}")
    for d in range(n_bins):
        region = "buffer" if d < buf else "inner"
        print(f"{d:>4} {mean_sq[d]:>12.4e} {mean_sq_zero[d]:>12.4e} "
              f"{mean_sq_lf[d]:>12.4e} {rel_to_zero[d]:>9.3f} "
              f"{int(count[d]):>10} {region:>8}")

    # Aggregate: inner-cube means
    inner_mask = np.arange(n_bins) >= buf
    inner_w = count * inner_mask
    if inner_w.sum() > 0:
        inner_model = float((sum_sq      * inner_mask).sum() / inner_w.sum())
        inner_zero  = float((sum_sq_zero * inner_mask).sum() / inner_w.sum())
        inner_rel   = inner_model / inner_zero if inner_zero > 0 else float("nan")
        print()
        print(f"[boundary] inner-cube aggregate:  "
              f"model_mse={inner_model:.4e}  "
              f"zero_mse={inner_zero:.4e}  "
              f"rel_to_zero={inner_rel:.3f}  "
              f"({'beats' if inner_rel < 1.0 else 'WORSE THAN'} predict-zero)")


if __name__ == "__main__":
    main()
