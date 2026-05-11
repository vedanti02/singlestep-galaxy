"""Aggregate per-run training and diagnostic metrics for side-by-side comparison.

Walks ``runs/*/`` directories and prints, per run:
  - epochs trained, best val loss + epoch
  - boundary diagnostic rel_to_zero (inner-cube aggregate) if available
  - cfg highlights (n_blocks, fields, lr)

Usage:
    python compare_runs.py
    python compare_runs.py --runs_dir runs --pattern 'pvfm_*'
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def _read_log(run_dir: Path) -> Optional[dict]:
    p = run_dir / "log.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_boundary(run_dir: Path) -> Optional[float]:
    """Return the latest inner-cube rel_to_zero from any boundary CSV under run_dir."""
    candidates = sorted(run_dir.glob("**/boundary_error.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        with open(candidates[0]) as f:
            rows = list(csv.DictReader(f))
        # weighted mean of (rel_to_zero) using count, restricted to inner bins.
        # Without knowing buf at this layer, treat all dist >= 4 as "inner"
        # (matches the default crop_overlap=8 setting).
        total_err = total_cnt = total_zero = 0.0
        for r in rows:
            d = int(r["distance_from_edge"])
            if d < 4:
                continue
            n = float(r["count"])
            total_err += float(r["mean_sq_error"]) * n
            total_zero += float(r["baseline_zero"]) * n
            total_cnt += n
        if total_cnt == 0 or total_zero == 0:
            return None
        return (total_err / total_cnt) / (total_zero / total_cnt)
    except (OSError, KeyError, ValueError):
        return None


def _read_ckpt_cfg(run_dir: Path) -> Optional[dict]:
    """Read the cfg dict from any checkpoint in run_dir without loading weights."""
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    try:
        import torch
        payload = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        return payload.get("cfg")
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--pattern", type=str, default="pvfm_*")
    args = p.parse_args()

    rows = []
    for d in sorted(Path(args.runs_dir).glob(args.pattern)):
        if not d.is_dir():
            continue
        log = _read_log(d)
        if log is None or not log.get("val"):
            continue
        val = log["val"]
        epochs = len(val)
        best_idx = min(range(epochs), key=lambda i: val[i]["loss"])
        best = val[best_idx]
        cfg = _read_ckpt_cfg(d) or {}
        m = cfg.get("model", {})
        dat = cfg.get("data", {})
        opt = cfg.get("optim", {})
        rel = _read_boundary(d)
        rows.append({
            "run":          d.name,
            "epochs":       epochs,
            "best_epoch":   best_idx,
            "best_val":     best["loss"],
            "best_val_pt":  best["pt_loss"],
            "rel_to_zero":  rel,
            "fields":       ",".join(dat.get("fields", ["disp"])),
            "n_blocks":     m.get("n_blocks", "?"),
            "base_voxel":   m.get("base_voxel", "?"),
            "base_point":   m.get("base_point", "?"),
            "lr":           opt.get("lr", "?"),
        })

    if not rows:
        print(f"[compare_runs] no runs with logs under {args.runs_dir}/{args.pattern}")
        return

    rows.sort(key=lambda r: (r["rel_to_zero"] if r["rel_to_zero"] is not None else 1e9,
                             r["best_val"]))

    cols = ["run", "epochs", "best_epoch", "best_val", "best_val_pt",
            "rel_to_zero", "fields", "n_blocks", "base_voxel", "base_point", "lr"]
    widths = {c: max(len(c), max(len(str(r[c] if r[c] is not None else "-"))
                                 for r in rows)) for c in cols}
    fmt = "  ".join(f"{{:<{widths[c]}}}" for c in cols)
    print(fmt.format(*cols))
    print(fmt.format(*("-" * widths[c] for c in cols)))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        print(fmt.format(*cells))


if __name__ == "__main__":
    main()
